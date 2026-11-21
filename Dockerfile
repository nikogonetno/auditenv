# Offline verification image. Nothing is pip-installed: the project has no
# runtime dependencies, so the build needs only the base image.
# Pinned by digest as well as by tag (QA L10): the digest CI resolved the tag to on 21.11.
FROM python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . .

# Privilege separation (QA risk R-03). The runner, env and grader run as root; every policy
# process is dropped to the unprivileged user auditpolicy (auditenv/runner.py). That user can
# read the code, but no private/ directory, cannot write under /app and cannot signal the
# runner. umask 077 keeps trajectories, grader logs and generated episodes private at runtime.
RUN useradd --uid 10002 --no-create-home --shell /usr/sbin/nologin auditpolicy \
 && chown -R root:root /app \
 && chmod -R go-w /app \
 && find /app -type d -name private -exec chmod -R go-rwx {} +

CMD ["sh", "-c", "umask 077 && exec sh scripts/verify.sh"]
