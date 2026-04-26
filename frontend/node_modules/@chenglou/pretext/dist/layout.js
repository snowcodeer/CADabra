// Text measurement for browser environments using canvas measureText.
//
// Problem: DOM-based text measurement (getBoundingClientRect, offsetHeight)
// forces synchronous layout reflow. When components independently measure text,
// each measurement triggers a reflow of the entire document. This creates
// read/write interleaving that can cost 30ms+ per frame for 500 text blocks.
//
// Solution: two-phase measurement centered around canvas measureText.
//   prepare(text, font) — segments text via Intl.Segmenter, measures each word
//     via canvas, caches widths, and does one cached DOM calibration read per
//     font when emoji correction is needed. Call once when text first appears.
//   layout(prepared, maxWidth, lineHeight) — walks cached word widths with pure
//     arithmetic to count lines and compute height. Call on every resize.
//     ~0.0002ms per text.
//
// i18n: Intl.Segmenter handles CJK (per-character breaking), Thai, Arabic, etc.
//   Bidi: simplified rich-path metadata for mixed LTR/RTL custom rendering.
//   Punctuation merging: "better." measured as one unit (matches CSS behavior).
//   Trailing whitespace: hangs past line edge without triggering breaks (CSS behavior).
//   overflow-wrap: pre-measured grapheme widths enable character-level word breaking.
//
// Emoji correction: Chrome/Firefox canvas measures emoji wider than DOM at font
//   sizes <24px on macOS (Apple Color Emoji). The inflation is constant per emoji
//   grapheme at a given size, font-independent. Auto-detected by comparing canvas
//   vs actual DOM emoji width (one cached DOM read per font). Safari canvas and
//   DOM agree (both wider than fontSize), so correction = 0 there.
//
// Limitations:
//   - system-ui font: canvas resolves to different optical variants than DOM on macOS.
//     Use named fonts (Helvetica, Inter, etc.) for guaranteed accuracy.
//     See RESEARCH.md "Discovery: system-ui font resolution mismatch".
//
// Based on Sebastian Markbage's text-layout research (github.com/chenglou/text-layout).
import { computeSegmentLevels } from './bidi.js';
import { analyzeText, canContinueKeepAllTextRun, clearAnalysisCaches, endsWithClosingQuote, isCJK, isNumericRunSegment, kinsokuEnd, kinsokuStart, leftStickyPunctuation, setAnalysisLocale, } from './analysis.js';
import { clearMeasurementCaches, getCorrectedSegmentWidth, getSegmentBreakableFitAdvances, getEngineProfile, getFontMeasurementState, getSegmentMetrics, textMayContainEmoji, } from './measurement.js';
import { countPreparedLines, measurePreparedLineGeometry, normalizeLineStart, stepPreparedLineGeometry, walkPreparedLinesRaw, } from './line-break.js';
import { buildLineTextFromRange, clearLineTextCaches, getLineTextCache, } from './line-text.js';
let sharedGraphemeSegmenter = null;
function getSharedGraphemeSegmenter() {
    if (sharedGraphemeSegmenter === null) {
        sharedGraphemeSegmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
    }
    return sharedGraphemeSegmenter;
}
// --- Public API ---
function createEmptyPrepared(includeSegments) {
    if (includeSegments) {
        return {
            widths: [],
            lineEndFitAdvances: [],
            lineEndPaintAdvances: [],
            kinds: [],
            simpleLineWalkFastPath: true,
            segLevels: null,
            breakableFitAdvances: [],
            letterSpacing: 0,
            spacingGraphemeCounts: [],
            discretionaryHyphenWidth: 0,
            tabStopAdvance: 0,
            chunks: [],
            segments: [],
        };
    }
    return {
        widths: [],
        lineEndFitAdvances: [],
        lineEndPaintAdvances: [],
        kinds: [],
        simpleLineWalkFastPath: true,
        segLevels: null,
        breakableFitAdvances: [],
        letterSpacing: 0,
        spacingGraphemeCounts: [],
        discretionaryHyphenWidth: 0,
        tabStopAdvance: 0,
        chunks: [],
    };
}
function buildBaseCjkUnits(segText, engineProfile) {
    const units = [];
    let unitParts = [];
    let unitStart = 0;
    let unitContainsCJK = false;
    let unitEndsWithClosingQuote = false;
    let unitIsSingleKinsokuEnd = false;
    function pushUnit() {
        if (unitParts.length === 0)
            return;
        units.push({
            text: unitParts.length === 1 ? unitParts[0] : unitParts.join(''),
            start: unitStart,
        });
        unitParts = [];
        unitContainsCJK = false;
        unitEndsWithClosingQuote = false;
        unitIsSingleKinsokuEnd = false;
    }
    function startUnit(grapheme, start, graphemeContainsCJK) {
        unitParts = [grapheme];
        unitStart = start;
        unitContainsCJK = graphemeContainsCJK;
        unitEndsWithClosingQuote = endsWithClosingQuote(grapheme);
        unitIsSingleKinsokuEnd = kinsokuEnd.has(grapheme);
    }
    function appendToUnit(grapheme, graphemeContainsCJK) {
        unitParts.push(grapheme);
        unitContainsCJK = unitContainsCJK || graphemeContainsCJK;
        const graphemeEndsWithClosingQuote = endsWithClosingQuote(grapheme);
        if (grapheme.length === 1 && leftStickyPunctuation.has(grapheme)) {
            unitEndsWithClosingQuote = unitEndsWithClosingQuote || graphemeEndsWithClosingQuote;
        }
        else {
            unitEndsWithClosingQuote = graphemeEndsWithClosingQuote;
        }
        unitIsSingleKinsokuEnd = false;
    }
    for (const gs of getSharedGraphemeSegmenter().segment(segText)) {
        const grapheme = gs.segment;
        const graphemeContainsCJK = isCJK(grapheme);
        if (unitParts.length === 0) {
            startUnit(grapheme, gs.index, graphemeContainsCJK);
            continue;
        }
        if (unitIsSingleKinsokuEnd ||
            kinsokuStart.has(grapheme) ||
            leftStickyPunctuation.has(grapheme) ||
            (engineProfile.carryCJKAfterClosingQuote &&
                graphemeContainsCJK &&
                unitEndsWithClosingQuote)) {
            appendToUnit(grapheme, graphemeContainsCJK);
            continue;
        }
        if (!unitContainsCJK && !graphemeContainsCJK) {
            appendToUnit(grapheme, graphemeContainsCJK);
            continue;
        }
        pushUnit();
        startUnit(grapheme, gs.index, graphemeContainsCJK);
    }
    pushUnit();
    return units;
}
function mergeKeepAllTextUnits(units) {
    if (units.length <= 1)
        return units;
    const merged = [];
    let currentTextParts = [units[0].text];
    let currentStart = units[0].start;
    let currentContainsCJK = isCJK(units[0].text);
    let currentCanContinue = canContinueKeepAllTextRun(units[0].text);
    function flushCurrent() {
        merged.push({
            text: currentTextParts.length === 1 ? currentTextParts[0] : currentTextParts.join(''),
            start: currentStart,
        });
    }
    for (let i = 1; i < units.length; i++) {
        const next = units[i];
        const nextContainsCJK = isCJK(next.text);
        const nextCanContinue = canContinueKeepAllTextRun(next.text);
        if (currentContainsCJK && currentCanContinue) {
            currentTextParts.push(next.text);
            currentContainsCJK = currentContainsCJK || nextContainsCJK;
            currentCanContinue = nextCanContinue;
            continue;
        }
        flushCurrent();
        currentTextParts = [next.text];
        currentStart = next.start;
        currentContainsCJK = nextContainsCJK;
        currentCanContinue = nextCanContinue;
    }
    flushCurrent();
    return merged;
}
function countRenderedSpacingGraphemes(text, kind) {
    if (kind === 'zero-width-break' ||
        kind === 'soft-hyphen' ||
        kind === 'hard-break') {
        return 0;
    }
    if (kind === 'tab')
        return 1;
    let count = 0;
    const graphemeSegmenter = getSharedGraphemeSegmenter();
    for (const _ of graphemeSegmenter.segment(text))
        count++;
    return count;
}
function addInternalLetterSpacing(width, graphemeCount, letterSpacing) {
    return graphemeCount > 1 ? width + (graphemeCount - 1) * letterSpacing : width;
}
function measureAnalysis(analysis, font, includeSegments, wordBreak, letterSpacing) {
    const engineProfile = getEngineProfile();
    const { cache, emojiCorrection } = getFontMeasurementState(font, textMayContainEmoji(analysis.normalized));
    const discretionaryHyphenWidth = getCorrectedSegmentWidth('-', getSegmentMetrics('-', cache), emojiCorrection) +
        (letterSpacing === 0 ? 0 : letterSpacing);
    const spaceWidth = getCorrectedSegmentWidth(' ', getSegmentMetrics(' ', cache), emojiCorrection);
    const tabStopAdvance = spaceWidth * 8;
    const hasLetterSpacing = letterSpacing !== 0;
    if (analysis.len === 0)
        return createEmptyPrepared(includeSegments);
    const widths = [];
    const lineEndFitAdvances = [];
    const lineEndPaintAdvances = [];
    const kinds = [];
    let simpleLineWalkFastPath = analysis.chunks.length <= 1 && !hasLetterSpacing;
    const segStarts = includeSegments ? [] : null;
    const breakableFitAdvances = [];
    const spacingGraphemeCounts = [];
    const segments = includeSegments ? [] : null;
    const preparedStartByAnalysisIndex = Array.from({ length: analysis.len });
    function pushMeasuredSegment(text, width, lineEndFitAdvance, lineEndPaintAdvance, kind, start, breakableFitAdvance, spacingGraphemeCount) {
        if (kind !== 'text' && kind !== 'space' && kind !== 'zero-width-break') {
            simpleLineWalkFastPath = false;
        }
        widths.push(width);
        lineEndFitAdvances.push(lineEndFitAdvance);
        lineEndPaintAdvances.push(lineEndPaintAdvance);
        kinds.push(kind);
        segStarts?.push(start);
        breakableFitAdvances.push(breakableFitAdvance);
        if (hasLetterSpacing)
            spacingGraphemeCounts.push(spacingGraphemeCount);
        if (segments !== null)
            segments.push(text);
    }
    function pushMeasuredTextSegment(text, kind, start, wordLike, allowOverflowBreaks) {
        const textMetrics = getSegmentMetrics(text, cache);
        const spacingGraphemeCount = hasLetterSpacing
            ? countRenderedSpacingGraphemes(text, kind)
            : 0;
        const width = addInternalLetterSpacing(getCorrectedSegmentWidth(text, textMetrics, emojiCorrection), spacingGraphemeCount, letterSpacing);
        const baseLineEndFitAdvance = kind === 'space' || kind === 'preserved-space' || kind === 'zero-width-break'
            ? 0
            : width;
        const lineEndFitAdvance = baseLineEndFitAdvance === 0
            ? 0
            : baseLineEndFitAdvance + (spacingGraphemeCount > 0 ? letterSpacing : 0);
        const lineEndPaintAdvance = kind === 'space' || kind === 'zero-width-break'
            ? 0
            : width;
        if (allowOverflowBreaks && wordLike && text.length > 1) {
            let fitMode = 'sum-graphemes';
            if (letterSpacing !== 0) {
                fitMode = 'segment-prefixes';
            }
            else if (isNumericRunSegment(text)) {
                fitMode = 'pair-context';
            }
            else if (engineProfile.preferPrefixWidthsForBreakableRuns) {
                fitMode = 'segment-prefixes';
            }
            const fitAdvances = getSegmentBreakableFitAdvances(text, textMetrics, cache, emojiCorrection, fitMode);
            pushMeasuredSegment(text, width, lineEndFitAdvance, lineEndPaintAdvance, kind, start, fitAdvances, spacingGraphemeCount);
            return;
        }
        pushMeasuredSegment(text, width, lineEndFitAdvance, lineEndPaintAdvance, kind, start, null, spacingGraphemeCount);
    }
    for (let mi = 0; mi < analysis.len; mi++) {
        preparedStartByAnalysisIndex[mi] = widths.length;
        const segText = analysis.texts[mi];
        const segWordLike = analysis.isWordLike[mi];
        const segKind = analysis.kinds[mi];
        const segStart = analysis.starts[mi];
        if (segKind === 'soft-hyphen') {
            pushMeasuredSegment(segText, 0, discretionaryHyphenWidth, discretionaryHyphenWidth, segKind, segStart, null, 0);
            continue;
        }
        if (segKind === 'hard-break') {
            pushMeasuredSegment(segText, 0, 0, 0, segKind, segStart, null, 0);
            continue;
        }
        if (segKind === 'tab') {
            pushMeasuredSegment(segText, 0, 0, 0, segKind, segStart, null, hasLetterSpacing ? countRenderedSpacingGraphemes(segText, segKind) : 0);
            continue;
        }
        const segMetrics = getSegmentMetrics(segText, cache);
        if (segKind === 'text' && segMetrics.containsCJK) {
            const baseUnits = buildBaseCjkUnits(segText, engineProfile);
            const measuredUnits = wordBreak === 'keep-all'
                ? mergeKeepAllTextUnits(baseUnits)
                : baseUnits;
            for (let i = 0; i < measuredUnits.length; i++) {
                const unit = measuredUnits[i];
                pushMeasuredTextSegment(unit.text, 'text', segStart + unit.start, segWordLike, wordBreak === 'keep-all' || !isCJK(unit.text));
            }
            continue;
        }
        pushMeasuredTextSegment(segText, segKind, segStart, segWordLike, true);
    }
    const chunks = mapAnalysisChunksToPreparedChunks(analysis.chunks, preparedStartByAnalysisIndex, widths.length);
    const segLevels = segStarts === null ? null : computeSegmentLevels(analysis.normalized, segStarts);
    if (segments !== null) {
        return {
            widths,
            lineEndFitAdvances,
            lineEndPaintAdvances,
            kinds,
            simpleLineWalkFastPath,
            segLevels,
            breakableFitAdvances,
            letterSpacing,
            spacingGraphemeCounts,
            discretionaryHyphenWidth,
            tabStopAdvance,
            chunks,
            segments,
        };
    }
    return {
        widths,
        lineEndFitAdvances,
        lineEndPaintAdvances,
        kinds,
        simpleLineWalkFastPath,
        segLevels,
        breakableFitAdvances,
        letterSpacing,
        spacingGraphemeCounts,
        discretionaryHyphenWidth,
        tabStopAdvance,
        chunks,
    };
}
function mapAnalysisChunksToPreparedChunks(chunks, preparedStartByAnalysisIndex, preparedEndSegmentIndex) {
    const preparedChunks = [];
    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const startSegmentIndex = chunk.startSegmentIndex < preparedStartByAnalysisIndex.length
            ? preparedStartByAnalysisIndex[chunk.startSegmentIndex]
            : preparedEndSegmentIndex;
        const endSegmentIndex = chunk.endSegmentIndex < preparedStartByAnalysisIndex.length
            ? preparedStartByAnalysisIndex[chunk.endSegmentIndex]
            : preparedEndSegmentIndex;
        const consumedEndSegmentIndex = chunk.consumedEndSegmentIndex < preparedStartByAnalysisIndex.length
            ? preparedStartByAnalysisIndex[chunk.consumedEndSegmentIndex]
            : preparedEndSegmentIndex;
        preparedChunks.push({
            startSegmentIndex,
            endSegmentIndex,
            consumedEndSegmentIndex,
        });
    }
    return preparedChunks;
}
function prepareInternal(text, font, includeSegments, options) {
    const wordBreak = options?.wordBreak ?? 'normal';
    const letterSpacing = options?.letterSpacing ?? 0;
    const analysis = analyzeText(text, getEngineProfile(), options?.whiteSpace, wordBreak);
    return measureAnalysis(analysis, font, includeSegments, wordBreak, letterSpacing);
}
// Prepare text for layout. Segments the text, measures each segment via canvas,
// and stores the widths for fast relayout at any width. Call once per text block
// (e.g. when a comment first appears). The result is width-independent — the
// same PreparedText can be laid out at any maxWidth and lineHeight via layout().
//
// Steps:
//   1. Normalize collapsible whitespace (CSS white-space: normal behavior)
//   2. Segment via Intl.Segmenter (handles CJK, Thai, etc.)
//   3. Merge punctuation into preceding word ("better." as one unit)
//   4. Split CJK words into individual graphemes (per-character line breaks)
//   5. Measure each segment via canvas measureText, cache by (segment, font)
//   6. Pre-measure graphemes of long words (for overflow-wrap: break-word)
//   7. Correct emoji canvas inflation (auto-detected per font size)
//   8. Optionally compute rich-path bidi metadata for custom renderers
export function prepare(text, font, options) {
    return prepareInternal(text, font, false, options);
}
// Rich variant used by callers that need enough information to render the
// laid-out lines themselves.
export function prepareWithSegments(text, font, options) {
    return prepareInternal(text, font, true, options);
}
function getInternalPrepared(prepared) {
    return prepared;
}
// Layout prepared text at a given max width and caller-provided lineHeight.
// Pure arithmetic on cached widths — no canvas calls, no DOM reads, no string
// operations, no allocations.
// ~0.0002ms per text block. Call on every resize.
//
// Line breaking rules (matching CSS white-space: normal + overflow-wrap: break-word):
//   - Break before any non-space segment that would overflow the line
//   - Trailing whitespace hangs past the line edge (doesn't trigger breaks)
//   - Segments wider than maxWidth are broken at grapheme boundaries
export function layout(prepared, maxWidth, lineHeight) {
    // Keep the resize hot path specialized. `layoutWithLines()` shares the same
    // break semantics but also tracks line ranges; the extra bookkeeping is too
    // expensive to pay on every hot-path `layout()` call.
    const lineCount = countPreparedLines(getInternalPrepared(prepared), maxWidth);
    return { lineCount, height: lineCount * lineHeight };
}
function createLayoutLine(prepared, cache, width, startSegmentIndex, startGraphemeIndex, endSegmentIndex, endGraphemeIndex) {
    return {
        text: buildLineTextFromRange(prepared, cache, startSegmentIndex, startGraphemeIndex, endSegmentIndex, endGraphemeIndex),
        width,
        start: {
            segmentIndex: startSegmentIndex,
            graphemeIndex: startGraphemeIndex,
        },
        end: {
            segmentIndex: endSegmentIndex,
            graphemeIndex: endGraphemeIndex,
        },
    };
}
function createLayoutLineRange(width, startSegmentIndex, startGraphemeIndex, endSegmentIndex, endGraphemeIndex) {
    return {
        width,
        start: {
            segmentIndex: startSegmentIndex,
            graphemeIndex: startGraphemeIndex,
        },
        end: {
            segmentIndex: endSegmentIndex,
            graphemeIndex: endGraphemeIndex,
        },
    };
}
export function materializeLineRange(prepared, line) {
    return createLayoutLine(prepared, getLineTextCache(prepared), line.width, line.start.segmentIndex, line.start.graphemeIndex, line.end.segmentIndex, line.end.graphemeIndex);
}
// Batch low-level line-range pass. This is the non-materializing counterpart
// to layoutWithLines(), useful for shrinkwrap and other aggregate stats work.
export function walkLineRanges(prepared, maxWidth, onLine) {
    if (prepared.widths.length === 0)
        return 0;
    return walkPreparedLinesRaw(getInternalPrepared(prepared), maxWidth, (width, startSegmentIndex, startGraphemeIndex, endSegmentIndex, endGraphemeIndex) => {
        onLine(createLayoutLineRange(width, startSegmentIndex, startGraphemeIndex, endSegmentIndex, endGraphemeIndex));
    });
}
export function measureLineStats(prepared, maxWidth) {
    return measurePreparedLineGeometry(getInternalPrepared(prepared), maxWidth);
}
// Intrinsic-width helper for rich/userland layout work. This asks "how wide is
// the prepared text when container width is not the thing forcing wraps?".
// Explicit hard breaks still count, so this returns the widest forced line.
export function measureNaturalWidth(prepared) {
    let maxWidth = 0;
    walkPreparedLinesRaw(getInternalPrepared(prepared), Number.POSITIVE_INFINITY, width => {
        if (width > maxWidth)
            maxWidth = width;
    });
    return maxWidth;
}
export function layoutNextLine(prepared, start, maxWidth) {
    const internal = getInternalPrepared(prepared);
    const normalizedStart = normalizeLineStart(internal, start);
    if (normalizedStart === null)
        return null;
    const end = {
        segmentIndex: normalizedStart.segmentIndex,
        graphemeIndex: normalizedStart.graphemeIndex,
    };
    const width = stepPreparedLineGeometry(internal, end, maxWidth);
    if (width === null)
        return null;
    return createLayoutLine(prepared, getLineTextCache(prepared), width, normalizedStart.segmentIndex, normalizedStart.graphemeIndex, end.segmentIndex, end.graphemeIndex);
}
export function layoutNextLineRange(prepared, start, maxWidth) {
    const internal = getInternalPrepared(prepared);
    const normalizedStart = normalizeLineStart(internal, start);
    if (normalizedStart === null)
        return null;
    const end = {
        segmentIndex: normalizedStart.segmentIndex,
        graphemeIndex: normalizedStart.graphemeIndex,
    };
    const width = stepPreparedLineGeometry(internal, end, maxWidth);
    if (width === null)
        return null;
    return createLayoutLineRange(width, normalizedStart.segmentIndex, normalizedStart.graphemeIndex, end.segmentIndex, end.graphemeIndex);
}
// Rich layout API for callers that want the actual line contents and widths.
// Caller still supplies lineHeight at layout time. Mirrors layout()'s break
// decisions, but keeps extra per-line bookkeeping so it should stay off the
// resize hot path.
export function layoutWithLines(prepared, maxWidth, lineHeight) {
    const lines = [];
    if (prepared.widths.length === 0)
        return { lineCount: 0, height: 0, lines };
    const graphemeCache = getLineTextCache(prepared);
    const lineCount = walkPreparedLinesRaw(getInternalPrepared(prepared), maxWidth, (width, startSegmentIndex, startGraphemeIndex, endSegmentIndex, endGraphemeIndex) => {
        lines.push(createLayoutLine(prepared, graphemeCache, width, startSegmentIndex, startGraphemeIndex, endSegmentIndex, endGraphemeIndex));
    });
    return { lineCount, height: lineCount * lineHeight, lines };
}
export function clearCache() {
    clearAnalysisCaches();
    sharedGraphemeSegmenter = null;
    clearLineTextCaches();
    clearMeasurementCaches();
}
export function setLocale(locale) {
    setAnalysisLocale(locale);
    clearCache();
}
