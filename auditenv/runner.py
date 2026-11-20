"""Run one episode with a policy in a separate process or in-process (case §04, §06, §09; role B).

The policy talks to the environment only through JSON lines (UTF-8, one object per line):

    runner -> policy   {"observation": obs, "info": info}                    after reset
                       {"observation", "reward", "terminated", "truncated", "info"}  after every step
    policy -> runner   one action object per line

Every policy line goes through `auditenv.server.decode_message`. A line that does not decode,
or is longer than MAX_MESSAGE_BYTES, becomes `env.step(None)`: the env answers `bad_action` and
spends a step (PM decision on §8.4). The episode ends on terminated or truncated. The wall-clock
limit, or the policy closing its stdout, ends it through `env.truncate()`.

The policy process gets a fresh empty working directory and a minimal environment. It never
receives episode paths, files under private/, the seed of the generator or the grader log.
The real grade is `env.grader_log[-1]`; the runner returns only what the policy itself saw.
"""
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .server import MAX_MESSAGE_BYTES, BadMessage, decode_message

REPO_ROOT = Path(__file__).resolve().parent.parent
TIME_LIMIT_S = 60.0       # case §07: 60 seconds per episode
EXIT_GRACE_S = 2.0        # after the end: close stdin, wait this long, then kill
TOO_LARGE = "message too large"  # the BadMessage code decode_message gives for oversize input
MAX_RECORDED_ACTION = 1 << 16  # bigger actions are recorded as size and sha256 only
TRAJECTORY_KEYS = ("episode_id", "step", "action", "observation", "reward", "terminated", "truncated")

_EOF = object()           # the policy closed its stdout or exited
_OVERSIZE = object()      # a line longer than MAX_MESSAGE_BYTES, already discarded
_DISCARD_CHUNK = 1 << 16
_READ_QUEUE = 4           # lines read ahead of the env; the rest wait in the pipe (backpressure)


# JSON lines are split on "\n" only, but str.splitlines() also splits on these. They can only
# occur inside JSON strings, so escaping them keeps every reader on one record per line.
_LINE_BREAKS = {"\u2028": "\\u2028", "\u2029": "\\u2029", "\x85": "\\u0085"}


def encode_line(obj):
    """One protocol or trajectory line: sorted keys, UTF-8, trailing newline."""
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    for raw, escaped in _LINE_BREAKS.items():
        text = text.replace(raw, escaped)
    return text + "\n"


def decode_line(raw):
    """(action for env.step, action for the trajectory). Never raises."""
    if raw is _OVERSIZE:
        return None, {"bad_message": TOO_LARGE}
    try:
        action = decode_message(raw)
    except BadMessage as exc:
        return None, {"bad_message": str(exc)}
    return action, action


class _Transcript:
    """Steps the env, keeps the counters of the result and writes the trajectory lines."""

    def __init__(self, env, seed, config, trajectory):
        self.env = env
        self.trajectory = trajectory
        observation, info = env.reset(seed=seed, config=config)
        manifest = observation.get("manifest") if isinstance(observation, dict) else None
        self.episode_id = manifest.get("episode_id") if isinstance(manifest, dict) else None
        self.index = 0
        self.steps = 0
        self.seen_reward = 0.0
        self.terminated = self.truncated = False
        self._write(None, observation, 0.0, False, False)
        self.first_message = {"observation": observation, "info": info}

    @property
    def done(self):
        return self.terminated or self.truncated

    def step(self, action, recorded):
        self.steps += 1
        return self._record(recorded, self.env.step(action))

    def truncate(self, reason):
        return self._record({"truncate": reason}, self.env.truncate())

    def result(self, started):
        if self.trajectory is not None:
            self.trajectory.flush()
        return {
            "episode_id": self.episode_id,
            "steps": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "seen_reward": self.seen_reward,
            "wall_time_s": round(time.monotonic() - started, 3),
        }

    def _record(self, recorded, transition):
        observation, reward, terminated, truncated, info = transition
        self.index += 1
        self.seen_reward += reward
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        self._write(recorded, observation, reward, terminated, truncated)
        return {"observation": observation, "reward": reward, "terminated": terminated,
                "truncated": truncated, "info": info}

    def _write(self, action, observation, reward, terminated, truncated):
        if self.trajectory is None:
            return
        if action is not None:
            text = json.dumps(action, sort_keys=True, ensure_ascii=False)
            if len(text) > MAX_RECORDED_ACTION:  # one hostile episode must not write 100 MB
                digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
                action = {"oversize_action": {"chars": len(text), "sha256": digest}}
        line = {"episode_id": self.episode_id, "step": self.index, "action": action,
                "observation": observation, "reward": reward, "terminated": terminated, "truncated": truncated}
        self.trajectory.write(encode_line(line))


# ---------------------------------------------------------------------------- in-process
def run_inprocess_episode(policy_fn, env, seed, config, trajectory=None):
    """Run a trusted grader-side policy (the oracle, case §06) in this process.

    policy_fn(message) -> action, where message is exactly what a subprocess policy would read.
    Messages and actions make the same JSON round trip as over the pipe, so both runners give
    the same trajectory for the same actions. Exceptions of policy_fn propagate: it is our code.
    """
    started = time.monotonic()
    transcript = _Transcript(env, seed, config, trajectory)
    message = transcript.first_message
    while not transcript.done:
        action = policy_fn(json.loads(encode_line(message)))
        try:
            text = json.dumps(action, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError):  # not JSON at all: what a pipe could not carry
            decoded = None, {"bad_message": "invalid JSON"}
        else:
            decoded = decode_line(text)  # size, NaN, surrogates: the same checks as a pipe line
        message = transcript.step(*decoded)
    return transcript.result(started)


# ---------------------------------------------------------------------------- subprocess
def policy_environment():
    """The whole environment of a policy process. No episode paths, nothing from private/."""
    env = {"PATH": os.environ.get("PATH", os.defpath), "LANG": "C.UTF-8", "PYTHONPATH": str(REPO_ROOT)}
    if "PYTHONHASHSEED" in os.environ:
        env["PYTHONHASHSEED"] = os.environ["PYTHONHASHSEED"]
    if os.name == "nt" and "SYSTEMROOT" in os.environ:  # Windows Python cannot start without it
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return env


def _read_lines(stream, lines, stop):
    """Reader thread: policy stdout -> queue of raw lines, _OVERSIZE markers and a final _EOF."""
    def put(item):
        while not stop.is_set():
            try:
                lines.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    try:
        while True:
            line = stream.readline(MAX_MESSAGE_BYTES + 1)
            if not line:
                break
            if line.endswith(b"\n"):
                put(line[:-1])
            elif len(line) <= MAX_MESSAGE_BYTES:
                put(line)  # the last line, without a newline, before EOF
            else:
                while True:  # over the limit: drop the rest of the line without keeping it
                    chunk = stream.readline(_DISCARD_CHUNK)
                    if not chunk or chunk.endswith(b"\n"):
                        break
                put(_OVERSIZE)
    except (OSError, ValueError):
        pass
    finally:
        put(_EOF)


def _write_lines(stream, outbox):
    """Writer thread: a policy that never reads its stdin cannot block the runner."""
    try:
        while True:
            data = outbox.get()
            if data is None:
                break
            stream.write(data)
            stream.flush()
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _kill(proc):
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)  # the policy and anything it started
        else:
            proc.kill()
    except (OSError, ValueError):
        pass
    try:
        proc.wait(timeout=EXIT_GRACE_S)
    except subprocess.TimeoutExpired:
        pass


def run_subprocess_episode(policy_cmd, env, seed, config, time_limit=TIME_LIMIT_S, trajectory=None):
    """Run one episode against a policy process started from policy_cmd (an argv list).

    Returns {"episode_id", "steps", "terminated", "truncated", "seen_reward", "wall_time_s"}.
    `steps` counts policy lines passed to env.step; `seen_reward` is what the policy saw (0 in eval).
    A broken config (env.reset) or a command that cannot start (OSError) raises; nothing the
    policy does after it starts does.
    """
    transcript = _Transcript(env, seed, config, trajectory)
    started = time.monotonic()
    deadline = started + float(time_limit)
    # The policy may rename, replace or fill its working directory; cleanup must still not raise.
    parent = tempfile.mkdtemp(prefix="auditenv-policy-")
    workdir = os.path.join(parent, "work")
    os.mkdir(workdir)
    try:
        proc = subprocess.Popen(
            list(policy_cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=workdir, env=policy_environment(), start_new_session=hasattr(os, "killpg"),
        )
        lines, outbox, stop = queue.Queue(maxsize=_READ_QUEUE), queue.Queue(), threading.Event()
        reader = threading.Thread(target=_read_lines, args=(proc.stdout, lines, stop), daemon=True)
        writer = threading.Thread(target=_write_lines, args=(proc.stdin, outbox), daemon=True)
        reader.start()
        writer.start()
        try:
            outbox.put(encode_line(transcript.first_message).encode("utf-8"))
            while not transcript.done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    message = transcript.truncate("time_limit")
                else:
                    try:
                        raw = lines.get(timeout=remaining)
                    except queue.Empty:
                        continue
                    if raw is _EOF:
                        message = transcript.truncate("policy_closed")
                    else:
                        message = transcript.step(*decode_line(raw))
                outbox.put(encode_line(message).encode("utf-8"))
            result = transcript.result(started)
        finally:
            outbox.put(None)  # close the policy's stdin after the last message
            stop.set()        # the reader drains the pipe until EOF and keeps nothing
            try:
                proc.wait(timeout=EXIT_GRACE_S)
            except subprocess.TimeoutExpired:
                pass
            _kill(proc)
            writer.join(timeout=EXIT_GRACE_S)
            reader.join(timeout=EXIT_GRACE_S)
            if not reader.is_alive():
                proc.stdout.close()
    finally:
        try:
            shutil.rmtree(parent, ignore_errors=True)
        except Exception:  # RecursionError on a very deep tree the policy built
            pass
    return result
