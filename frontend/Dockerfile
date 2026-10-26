# syntax=docker/dockerfile:1
#
# Furina ML frontend -- the Vite bundle, served by nginx, which also reverse-
# proxies /api to the Django container so the whole app sits on one origin.

# --- build ---------------------------------------------------------------
# vite 7 requires node ^20.19 || >=22.12.
FROM node:22-alpine AS build

WORKDIR /app

# Lockfile first: `npm ci` only reruns when the dependencies actually change.
# --no-audit skips the advisory endpoint, which is slow and irrelevant to a
# lockfile install; the cache mount and retries make a flaky network survivable.
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund --fetch-timeout=600000 --fetch-retries=5

COPY . .
RUN npm run build

# --- notebooks: the JupyterLite preview -----------------------------------
# A static site in which Python runs in the visitor's browser. The course
# notebooks arrive as the `notebooks` build context (docker-compose.yml);
# without compose: docker build --build-context notebooks=../notebooks .
FROM python:3.13-slim AS lite

WORKDIR /lite
COPY jupyterlite/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --disable-pip-version-check -r requirements.txt
COPY --from=notebooks . /contents
RUN rm -rf /contents/tools /contents/README.md \
 && jupyter lite build --contents /contents --output-dir /site --no-sourcemaps

# --- runtime -------------------------------------------------------------
FROM nginx:1.29-alpine AS runtime

# The template, not the finished config: 10-furina-config.sh renders it at
# start-up so BACKEND_ORIGIN and API_BASE_URL are runtime settings.
COPY nginx.conf.template /etc/furina/nginx.conf.template
COPY nginx-jupyterhub.conf.template /etc/furina/nginx-jupyterhub.conf.template
COPY docker-entrypoint.d/10-furina-config.sh /docker-entrypoint.d/10-furina-config.sh
RUN chmod +x /docker-entrypoint.d/10-furina-config.sh \
 && rm -f /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist /usr/share/nginx/html
COPY --from=lite /site /usr/share/nginx/html/notebooks

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD wget --quiet --tries=1 --spider http://127.0.0.1/index.html || exit 1
