# Offline verification image. Nothing is pip-installed: the project has no
# runtime dependencies, so the build needs only the base image.
FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . .

# Never run as root. Later the grader gets its own user so a policy process
# cannot read private episode files (QA risk R-03).
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin runner
USER runner

CMD ["sh", "scripts/verify.sh"]
