#!/bin/sh
#
# Runs before nginx starts (the nginx image executes /docker-entrypoint.d/*.sh
# in order). Turns two environment variables into the two files that make this
# image redeployable without a rebuild:
#
#   API_BASE_URL    -> /usr/share/nginx/html/config.js, read by the SPA at
#                      runtime. Vite bakes import.meta.env at build time, so a
#                      runtime file is the only way to move the API without
#                      rebuilding the bundle.
#   BACKEND_ORIGIN  -> the proxy_pass target in the rendered nginx config.
set -eu

API_BASE_URL="${API_BASE_URL:-/api}"
BACKEND_ORIGIN="${BACKEND_ORIGIN:-backend:8000}"
MAX_UPLOAD_SIZE="${MAX_UPLOAD_SIZE:-50m}"
PROXY_TIMEOUT="${PROXY_TIMEOUT:-300s}"

# The public origin, used for the canonical link, the Open Graph URLs, the
# sitemap entries and the Sitemap: line in robots.txt. Baked into the bundle at
# build time; rewritten here so the same image can be deployed on another host
# without shipping someone else's domain in its <link rel="canonical">.
SITE_URL="${SITE_URL:-https://furina.nevatal.id}"
SITE_URL="${SITE_URL%/}"
BUILT_SITE_URL="https://furina.nevatal.id"

# nginx needs an explicit resolver to look the backend up per request rather
# than once at start-up. Inside Docker that is the embedded DNS server at
# 127.0.0.11, which is exactly what /etc/resolv.conf already points at -- so
# read it from there and the same image also works on Kubernetes or a plain
# `docker run --network host`.
RESOLVER="${DNS_RESOLVER:-}"
if [ -z "$RESOLVER" ]; then
    RESOLVER="$(awk '/^nameserver/ { print $2; exit }' /etc/resolv.conf 2>/dev/null || true)"
fi
RESOLVER="${RESOLVER:-127.0.0.11}"
# An IPv6 nameserver has to be bracketed in an nginx `resolver` directive.
case "$RESOLVER" in
    *:*) RESOLVER="[${RESOLVER}]" ;;
esac

# `http://host:port` is the natural thing to type, but nginx wants a bare
# `host:port` here. Accept either spelling.
BACKEND_ORIGIN="$(printf '%s' "$BACKEND_ORIGIN" | sed -e 's#^https\?://##' -e 's#/$##')"

TEMPLATE=/etc/furina/nginx.conf.template
TARGET=/etc/nginx/conf.d/default.conf

echo "[furina] backend upstream: ${BACKEND_ORIGIN} (via resolver ${RESOLVER})"
echo "[furina] SPA API base URL: ${API_BASE_URL}"

# `#` as the sed delimiter: none of these values may contain one, and the URLs
# are full of slashes.
sed \
    -e "s#__BACKEND_ORIGIN__#${BACKEND_ORIGIN}#g" \
    -e "s#__MAX_UPLOAD_SIZE__#${MAX_UPLOAD_SIZE}#g" \
    -e "s#__PROXY_TIMEOUT__#${PROXY_TIMEOUT}#g" \
    -e "s#__RESOLVER__#${RESOLVER}#g" \
    "$TEMPLATE" > "$TARGET"

cat > /usr/share/nginx/html/config.js <<CONFIG
// Generated at container start by 10-furina-config.sh -- do not edit.
window.__FURINA_API_BASE_URL__ = "${API_BASE_URL}";
CONFIG

if [ "$SITE_URL" != "$BUILT_SITE_URL" ]; then
    echo "[furina] rewriting canonical origin to ${SITE_URL}"
    for f in /usr/share/nginx/html/index.html \
             /usr/share/nginx/html/robots.txt \
             /usr/share/nginx/html/sitemap.xml; do
        [ -f "$f" ] || continue
        # `|` as the delimiter: the values are URLs, full of slashes.
        sed -i "s|${BUILT_SITE_URL}|${SITE_URL}|g" "$f"
    done
else
    echo "[furina] canonical origin: ${SITE_URL}"
fi
