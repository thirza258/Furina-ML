import { writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { COURSES, lessonPath } from "../src/data/courseCatalogue";

const __dirname = dirname(fileURLToPath(import.meta.url));
const TARGET_PATH = resolve(__dirname, "../public/sitemap.xml");
const SITE_URL = "https://furina.nevatal.id";

interface SitemapEntry {
  loc: string;
  changefreq: "always" | "hourly" | "daily" | "weekly" | "monthly" | "yearly" | "never";
  priority: string;
}

const entries: SitemapEntry[] = [
  { loc: `${SITE_URL}/`, changefreq: "weekly", priority: "1.0" },
  { loc: `${SITE_URL}/about`, changefreq: "monthly", priority: "0.7" },
  { loc: `${SITE_URL}/app`, changefreq: "weekly", priority: "0.8" },
  { loc: `${SITE_URL}/app/preprocessing`, changefreq: "monthly", priority: "0.8" },
  { loc: `${SITE_URL}/app/training`, changefreq: "monthly", priority: "0.8" },
  { loc: `${SITE_URL}/app/evaluation`, changefreq: "monthly", priority: "0.8" },
  { loc: `${SITE_URL}/app/dashboard`, changefreq: "monthly", priority: "0.8" },
  { loc: `${SITE_URL}/app/courses`, changefreq: "weekly", priority: "0.9" },
];

for (const course of COURSES) {
  entries.push({
    loc: `${SITE_URL}/app/courses/${course.id}`,
    changefreq: "monthly",
    priority: "0.8",
  });
  for (const lesson of course.lessons) {
    entries.push({
      loc: `${SITE_URL}${lessonPath(course, lesson)}`,
      changefreq: "monthly",
      priority: "0.7",
    });
  }
}

const xml = `<?xml version="1.0" encoding="UTF-8"?>
<!--
  Public pages, workbench tools (preprocessing, training, evaluation, dashboard),
  and interactive courses with all lesson topics.
-->
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${entries
  .map(
    (e) => `  <url>
    <loc>${e.loc}</loc>
    <changefreq>${e.changefreq}</changefreq>
    <priority>${e.priority}</priority>
  </url>`
  )
  .join("\n")}
</urlset>
`;

writeFileSync(TARGET_PATH, xml, "utf-8");
console.log(`Generated sitemap with ${entries.length} URLs at ${TARGET_PATH}`);
