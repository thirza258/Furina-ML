import { useEffect } from "react";

export interface PageSeoProps {
  title: string;
  description: string;
  path: string;
  jsonLd?: Record<string, unknown> | Array<Record<string, unknown>>;
}

const DEFAULT_SITE_URL = "https://furina.nevatal.id";

function setMeta(attributeName: "name" | "property", attributeValue: string, content: string) {
  let el = document.querySelector<HTMLMetaElement>(`meta[${attributeName}="${attributeValue}"]`);
  if (!el) {
    el = document.createElement("meta");
    el.setAttribute(attributeName, attributeValue);
    document.head.appendChild(el);
  }
  el.setAttribute("content", content);
}

function setCanonical(url: string) {
  let el = document.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  if (!el) {
    el = document.createElement("link");
    el.setAttribute("rel", "canonical");
    document.head.appendChild(el);
  }
  el.setAttribute("href", url);
}

function setJsonLd(data?: Record<string, unknown> | Array<Record<string, unknown>>) {
  const SCRIPT_ID = "route-json-ld";
  const existing = document.getElementById(SCRIPT_ID);
  if (!data) {
    existing?.remove();
    return;
  }
  let script = existing as HTMLScriptElement | null;
  if (!script) {
    script = document.createElement("script");
    script.id = SCRIPT_ID;
    script.type = "application/ld+json";
    document.head.appendChild(script);
  }
  script.textContent = JSON.stringify(data);
}

export function usePageSeo({ title, description, path, jsonLd }: PageSeoProps) {
  useEffect(() => {
    const origin =
      typeof window !== "undefined" && window.location?.origin
        ? window.location.origin
        : DEFAULT_SITE_URL;

    const fullTitle = title.includes("Furina ML") ? title : `${title} | Furina ML`;
    const cleanPath = path.startsWith("/") ? path : `/${path}`;
    const fullUrl = `${origin}${cleanPath}`;

    document.title = fullTitle;

    setMeta("name", "description", description);
    setMeta("property", "og:title", fullTitle);
    setMeta("property", "og:description", description);
    setMeta("property", "og:url", fullUrl);
    setMeta("property", "og:image", `${origin}/logo_furina.png`);
    setMeta("name", "twitter:title", fullTitle);
    setMeta("name", "twitter:description", description);
    setMeta("name", "twitter:image", `${origin}/logo_furina.png`);
    setCanonical(fullUrl);
    setJsonLd(jsonLd);
  }, [title, description, path, jsonLd]);
}
