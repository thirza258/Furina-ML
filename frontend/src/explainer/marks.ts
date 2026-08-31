/**
 * Typographic marks, pinned to their text presentation.
 *
 * U+26A0 WARNING SIGN defaults to *emoji* presentation in most browsers, so a
 * bare `⚠` in the markup renders as a yellow-and-black pictograph. That is
 * wrong here twice over: it does not belong on a page set in Georgia, and a
 * colour emoji is precisely the note this interface is trying not to strike.
 * U+FE0E (VARIATION SELECTOR-15) requests the monochrome glyph instead, so the
 * mark takes the surrounding ink colour like any other character.
 *
 * The other marks this app uses -- ▲ ● ✓ ✕ ◌ ▼ ↕ ↓ → -- have no emoji
 * presentation to suppress and are written literally where they appear.
 */

/** ⚠ in text presentation. Always paired with a word, never the only signal. */
export const WARNING_MARK = "⚠︎";
