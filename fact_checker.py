import streamlit as st
import streamlit.components.v1 as components
import os
import json
import time
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv

from excerpt_align import align_issue_excerpts, find_excerpt_ranges
from fact_check_request import build_fact_check_body, stream_fact_check

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Fact Checker", page_icon="🔍", layout="wide")

st.title("🔍 Fact Checker")
st.write("Enter text below to fact-check it for misleading, questionable, or incomplete information.")

# Initialize session state
if 'main_text_input' not in st.session_state:
    st.session_state.main_text_input = ""

if 'fact_check_results' not in st.session_state:
    st.session_state.fact_check_results = None

# Only models with a zero-data-retention endpoint on OpenRouter: the key runs under the
# "Company data: zero data retention" guardrail, which rejects every request to any other model.
FACT_CHECK_MODELS = [
    "openai/gpt-6.1-terra",
    "openai/gpt-6.1-sol",
    "openai/gpt-6-luna",
    "google/gemini-4-hydrogen",
    "anthropic/claude-sonnet-5.5",
    "openai/gpt-5",
]
MODEL_CHOICE = os.getenv("FACT_CHECK_MODEL", "openai/gpt-6.1-terra")

if MODEL_CHOICE not in FACT_CHECK_MODELS:
    st.error(f"FACT_CHECK_MODEL '{MODEL_CHOICE}' is not supported. Use one of: {', '.join(FACT_CHECK_MODELS)}.")
    st.stop()

# Paste helper to extract links from Google Docs
paste_helper = r"""
<script>
const namedEntities = {
  nbsp: " ",
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  shy: "",
  laquo: "«",
  raquo: "»",
  ndash: "–",
  mdash: "—",
  hellip: "…",
  bdquo: "„",
  ldquo: "“",
  rdquo: "”",
  sbquo: "‚",
  lsquo: "‘",
  rsquo: "’",
  middot: "·",
  bull: "•",
  copy: "©",
  reg: "®",
  trade: "™",
  euro: "€",
  deg: "°",
  sect: "§",
  acirc: "â",
  Acirc: "Â",
  icirc: "î",
  Icirc: "Î",
  abreve: "ă",
  Abreve: "Ă",
  scedil: "ş",
  Scedil: "Ş",
  tcedil: "ţ",
  Tcedil: "Ţ",
};

/** Elements that end a paragraph and leave an empty line before the next block. */
const gapTags = new Set(["p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "ul", "ol", "table", "hr"]);

/** Elements that only start a new line. */
const lineTags = new Set([
  "br", "div", "section", "article", "header", "footer", "main", "aside", "nav", "figure", "figcaption",
  "li", "tr", "dl", "dt", "dd", "tbody", "thead", "tfoot", "caption", "form", "address", "details", "summary",
  "fieldset", "center",
]);

const tokenPattern =
  /<!--[\s\S]*?-->|<[!?][^>]*>|<(\/?)([a-zA-Z][\w:.-]*)((?:"[^"]*"|'[^']*'|[^>"'])*)>|([^<]+|<)/g;

/** @param {string} value */
function decodeEntities(value) {
  return value.replace(/&(#x[0-9a-f]+|#\d+|[a-z][a-z0-9]*);/gi, (match, name) => {
    if (name[0] === "#") {
      const codePoint = name[1] === "x" || name[1] === "X" ? parseInt(name.slice(2), 16) : parseInt(name.slice(1), 10);
      const isSurrogate = codePoint >= 0xd800 && codePoint <= 0xdfff;
      return codePoint > 0 && codePoint <= 0x10ffff && !isSurrogate ? String.fromCodePoint(codePoint) : match;
    }
    return Object.hasOwn(namedEntities, name) ? namedEntities[name] : match;
  });
}

/**
 * @param {string} attributes
 * @param {string} name
 * @returns {string | null}
 */
function getAttribute(attributes, name) {
  const match = new RegExp(`\\s${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s"'>]+))`, "i").exec(attributes);
  if (!match) return null;
  return decodeEntities(match[1] ?? match[2] ?? match[3] ?? "").trim();
}

/**
 * Resolve an href to an absolute http(s) URL, unwrapping Google redirect links.
 * Returns null for in-page anchors, mailto:, tel:, javascript: and anything unresolvable.
 *
 * @param {string | null} href
 * @param {string | undefined} baseUrl
 * @returns {string | null}
 */
function resolveHref(href, baseUrl) {
  if (!href || href.startsWith("#")) return null;
  let url;
  try {
    url = new URL(href, baseUrl);
  } catch {
    return null;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return null;

  if (/(^|\.)google\.[a-z.]+$/i.test(url.hostname) && url.pathname === "/url") {
    const target = url.searchParams.get("q") ?? url.searchParams.get("url");
    if (target) {
      try {
        const real = new URL(target);
        if (real.protocol === "http:" || real.protocol === "https:") return real.href;
      } catch {
        return url.href;
      }
    }
  }
  return url.href;
}

/** @param {string} value */
function comparableUrl(value) {
  return value.trim().replace(/^https?:\/\//i, "").replace(/\/+$/, "").toLowerCase();
}

/** @param {string | null} value */
function isTinyDimension(value) {
  if (value === null) return false;
  const size = parseInt(value, 10);
  return Number.isFinite(size) && size <= 2;
}

/**
 * True when the clipboard HTML carries something plain text would lose:
 * an image or a link with an http(s) or relative href.
 *
 * @param {string} html
 * @returns {boolean}
 */
function hasLinkOrImage(html) {
  if (/<img\b/i.test(html)) return true;
  for (const match of html.matchAll(/<a\b[^>]*\shref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi)) {
    const href = (match[1] ?? match[2] ?? match[3] ?? "").trim();
    if (/^https?:/i.test(href)) return true;
    if (href !== "" && !/^(?:#|[a-z][a-z0-9+.-]*:)/i.test(href)) return true;
  }
  return false;
}

/**
 * Convert HTML to plain text that keeps links as `text (URL)` and images as `[Image: alt]`.
 *
 * @param {string} html
 * @param {string} [baseUrl] used to resolve relative hrefs
 * @returns {string}
 */
function htmlToText(html, baseUrl) {
  const source = html
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<!\[if !supportLists\]>[\s\S]*?<!\[endif\]>/gi, "- ")
    .replace(/<(script|style|noscript|template|head|svg|xml)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, "");

  /** @type {string[]} */
  const lines = [];
  let buffer = "";
  let pendingPrefix = "";
  let pendingGap = false;
  let listDepth = 0;
  let cellsInRow = 0;
  /** @type {{ url: string | null, text: string } | null} */
  let link = null;

  /** @param {string} line */
  function pushLine(line) {
    if (pendingGap && lines.length > 0) lines.push("");
    pendingGap = false;
    lines.push(line);
  }

  function newline() {
    const line = buffer.trim();
    buffer = "";
    if (line) pushLine(line);
  }

  function gap() {
    newline();
    pendingGap = true;
  }

  /** @param {string} piece */
  function put(piece) {
    if (buffer === "") {
      buffer = pendingPrefix + piece;
      pendingPrefix = "";
    } else {
      buffer += piece;
    }
  }

  /** @param {string} piece */
  function append(piece) {
    put(buffer === "" || buffer.endsWith(" ") ? piece : ` ${piece}`);
  }

  /** @param {string} raw */
  function addText(raw) {
    let piece = decodeEntities(raw).replace(/\s+/g, " ");
    if (piece.startsWith(" ") && (buffer === "" || buffer.endsWith(" "))) piece = piece.slice(1);
    if (!piece) return;
    put(piece);
    if (link) link.text += piece;
  }

  function closeLink() {
    if (!link) return;
    const { url, text } = link;
    link = null;
    if (!url) return;

    const label = text.trim();
    if (!label) {
      append(url);
    } else if (comparableUrl(label) !== comparableUrl(url)) {
      const trailingSpace = text.endsWith(" ") && buffer.endsWith(" ");
      buffer = buffer.trimEnd();
      append(`(${url})`);
      if (trailingSpace) buffer += " ";
    }
  }

  /**
   * @param {string} attributes
   */
  function addImage(attributes) {
    const src = getAttribute(attributes, "src") ?? "";
    const isTinyData = /^data:/i.test(src) && src.length < 200;
    if (isTinyDimension(getAttribute(attributes, "width")) || isTinyDimension(getAttribute(attributes, "height")) || isTinyData) {
      return;
    }
    const alt = (getAttribute(attributes, "alt") ?? "").replace(/\s+/g, " ").trim();
    newline();
    pushLine(alt ? `[Image: ${alt}]` : "[Image]");
  }

  for (const match of source.matchAll(tokenPattern)) {
    const [, closing, rawName, attributes, text] = match;
    if (text !== undefined) {
      addText(text);
      continue;
    }
    if (!rawName) continue;

    const name = rawName.toLowerCase();
    const isClosing = closing === "/";

    if (name === "a") {
      closeLink();
      if (!isClosing) {
        link = { url: resolveHref(getAttribute(attributes, "href"), baseUrl), text: "" };
      }
      continue;
    }
    if (name === "img") {
      if (!isClosing) addImage(attributes);
      continue;
    }
    if (name === "ul" || name === "ol") {
      if (isClosing) listDepth = Math.max(0, listDepth - 1);
      if (listDepth > 0) newline();
      else gap();
      if (!isClosing) listDepth += 1;
      continue;
    }
    if (name === "li") {
      newline();
      pendingPrefix = isClosing ? "" : "- ";
      continue;
    }
    if (name === "td" || name === "th") {
      if (!isClosing && cellsInRow++ > 0) {
        buffer = buffer.trimEnd() + " | ";
      }
      continue;
    }
    if (name === "tr") {
      newline();
      cellsInRow = 0;
      continue;
    }
    if (gapTags.has(name)) {
      if (listDepth > 0 && name !== "table") newline();
      else gap();
      continue;
    }
    if (lineTags.has(name)) newline();
  }

  closeLink();
  newline();
  return lines.join("\n");
}

function attachPasteHandler() {
    const textareas = window.parent.document.querySelectorAll('textarea');

    textareas.forEach(textarea => {
        if (textarea.dataset.pasteHandlerAttached) return;
        textarea.dataset.pasteHandlerAttached = 'true';

        textarea.addEventListener('paste', function(e) {
            const clipboardData = e.clipboardData || window.clipboardData;
            const htmlData = clipboardData.getData('text/html');

            if (htmlData && hasLinkOrImage(htmlData)) {
                e.preventDefault();

                const processedText = htmlToText(htmlData);
                const start = textarea.selectionStart;
                const end = textarea.selectionEnd;
                const currentValue = textarea.value;
                const newValue = currentValue.substring(0, start) + processedText + currentValue.substring(end);

                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeInputValueSetter.call(textarea, newValue);

                const newPos = start + processedText.length;
                textarea.selectionStart = newPos;
                textarea.selectionEnd = newPos;

                textarea.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
                textarea.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
    });
}

attachPasteHandler();
setInterval(attachPasteHandler, 500);
</script>
"""

components.html(paste_helper, height=0)

st.text_area(
    "",
    height=300,
    placeholder="Paste your text here (links and images from Google Docs are kept automatically)...",
    key="main_text_input"
)

submit_button = st.button("Fact Check", type="primary")


def highlight_text(original_text, issues, show_misleading=True, show_incomplete=True, show_questionable=True):
    """Highlight problematic sections in the text with overlapping support"""
    if not issues:
        return original_text

    # Filter issues based on visibility settings
    filtered_issues = []
    for idx, issue in enumerate(issues):
        issue_type = issue.get('type', 'questionable')
        if issue_type == 'misleading' and not show_misleading:
            continue
        if issue_type == 'incomplete' and not show_incomplete:
            continue
        if issue_type == 'questionable' and not show_questionable:
            continue

        issue_with_idx = issue.copy()
        issue_with_idx['_original_index'] = idx
        filtered_issues.append(issue_with_idx)

    if not filtered_issues:
        return original_text

    colors = {
        'misleading': '#ffcccc',
        'questionable': '#fff4cc',
        'incomplete': '#cce5ff'
    }

    border_colors = {
        'misleading': '#ff0000',
        'questionable': '#ffa500',
        'incomplete': '#0066cc'
    }

    # Find all highlight positions
    highlight_ranges = []

    for issue in filtered_issues:
        excerpt = issue.get('excerpt', '').strip()
        if not excerpt:
            continue

        issue_type = issue.get('type', 'questionable')
        explanation = issue.get('issue', 'No explanation')
        sources = issue.get('sources', [])
        original_idx = issue.get('_original_index', 0)

        for start_pos, end_pos in find_excerpt_ranges(original_text, excerpt):
            highlight_ranges.append({
                'start': start_pos,
                'end': end_pos,
                'issue_index': original_idx,
                'issue_type': issue_type,
                'explanation': explanation,
                'sources': sources
            })

    highlight_ranges.sort(key=lambda x: x['start'])

    # Build segments with overlapping highlights
    position_issues = {}
    for hr in highlight_ranges:
        for pos in range(hr['start'], hr['end']):
            if pos not in position_issues:
                position_issues[pos] = []
            position_issues[pos].append(hr)

    positions = sorted(set([0, len(original_text)] +
                          [hr['start'] for hr in highlight_ranges] +
                          [hr['end'] for hr in highlight_ranges]))

    segments = []
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]
        active_issues = position_issues.get(start, [])

        segments.append({
            'start': start,
            'end': end,
            'text': original_text[start:end],
            'issues': active_issues
        })

    # Build HTML
    html_parts = []

    for segment in segments:
        if not segment['issues']:
            html_parts.append(segment['text'])
        else:
            issues_data = segment['issues']
            primary = issues_data[0]
            bg_color = colors.get(primary['issue_type'], '#fff4cc')

            # Build tooltip
            tooltip_parts = []
            all_issue_indices = []

            for idx, issue_data in enumerate(issues_data):
                all_issue_indices.append(str(issue_data['issue_index']))

                tooltip_parts.append(f"Issue #{issue_data['issue_index'] + 1} ({issue_data['issue_type'].title()}):")
                tooltip_parts.append(issue_data['explanation'])

                if issue_data['sources']:
                    tooltip_parts.append("\nSources:")
                    for src_idx, src in enumerate(issue_data['sources'], 1):
                        tooltip_parts.append(f"{src_idx}. {src}")

                if idx < len(issues_data) - 1:
                    tooltip_parts.append("\n---\n")

            tooltip_text = "\n".join(tooltip_parts)
            escaped_tooltip = tooltip_text.replace('"', '&quot;').replace("'", '&#39;')

            border_style = "2px solid transparent"
            if len(issues_data) > 1:
                border_colors_list = [border_colors.get(iss['issue_type'], '#333') for iss in issues_data[1:]]
                if border_colors_list:
                    border_style = f"2px solid {border_colors_list[0]}"

            issue_ids = ','.join(all_issue_indices)
            badge_text = str(primary['issue_index'] + 1) if len(issues_data) == 1 else str(len(issues_data))
            badge_type = "single" if len(issues_data) == 1 else "multiple"

            html_parts.append(f'''<mark
                class="fact-issue"
                data-issue-ids="{issue_ids}"
                data-tooltip="{escaped_tooltip}"
                data-badge-text="{badge_text}"
                data-badge-type="{badge_type}"
                style="background-color: {bg_color}; color: #000; padding: 2px 4px; border-radius: 3px; cursor: pointer; border: {border_style}; transition: all 0.2s ease;"
                >{segment['text']}</mark>''')

    return ''.join(html_parts)


# Process fact-check when button is clicked
if submit_button:
    st.session_state.fact_check_results = None

    current_text = st.session_state.main_text_input
    if not current_text.strip():
        st.error("Please enter some text to fact-check.")
    elif not os.getenv("OPENROUTER_API_KEY"):
        st.error("OpenRouter API key not found. Please set OPENROUTER_API_KEY in your .env file.")
    else:
        status_placeholder = st.empty()
        search_placeholder = st.empty()
        started_at = time.monotonic()
        searches_so_far = []
        status_placeholder.info("Performing deep research and fact-checking...")

        def show_search(search):
            searches_so_far.append(search)
            status_placeholder.info(f"Performing deep research and fact-checking... {len(searches_so_far)} web searches so far")
            with search_placeholder.container():
                with st.expander("🧠 Reasoning Summary & Web Search", expanded=True):
                    for number, done in enumerate(searches_so_far, 1):
                        st.markdown(f"**🔍 Web Search {number}:** {done['query']} ({len(done['sources'])} sources)")

        try:
            body = build_fact_check_body(MODEL_CHOICE, current_text, datetime.now(timezone.utc))
            run = stream_fact_check(os.getenv("OPENROUTER_API_KEY"), body, show_search)

            result_json = json.loads(run['text'])
            alignment = align_issue_excerpts(
                current_text, result_json.get('issues', []), MODEL_CHOICE, os.getenv("OPENROUTER_API_KEY")
            )
            if alignment['repair_error']:
                st.warning(f"Excerpt repair failed: {alignment['repair_error']}")

            reasoning_and_search_items = []
            if run['reasoning']:
                reasoning_and_search_items.append({'type': 'reasoning', 'text': run['reasoning']})
            reasoning_and_search_items += [{'type': 'web_search', **search} for search in run['searches']]

            repair_cost = (alignment['repair_usage'] or {}).get('cost') or 0
            status_placeholder.empty()
            search_placeholder.empty()
            st.success("Fact-check complete!")

            st.session_state.fact_check_results = {
                'issues': alignment['issues'],
                'all_sources': result_json.get('all_sources', []),
                'current_text': current_text,
                'reasoning_and_search_items': reasoning_and_search_items,
                'cost': (run['usage'].get('cost') or 0) + repair_cost,
                'search_count': len(run['searches']),
                'duration': time.monotonic() - started_at
            }

        except json.JSONDecodeError as e:
            st.error(f"Failed to parse fact-check results. Error: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            with st.expander("Show error details"):
                st.code(traceback.format_exc())

# Display results from session state
if st.session_state.fact_check_results is not None:
    results = st.session_state.fact_check_results
    issues = results['issues']
    all_sources = results['all_sources']
    current_text = results['current_text']
    reasoning_and_search_items = results['reasoning_and_search_items']

    # Display reasoning summary and web searches
    if reasoning_and_search_items:
        with st.expander("🧠 Reasoning Summary & Web Search", expanded=True):
            search_counter = 0
            for item in reasoning_and_search_items:
                if item['type'] == 'reasoning':
                    st.markdown(item['text'])
                    st.markdown("")

                elif item['type'] == 'web_search':
                    search_counter += 1
                    query = item['query']
                    sources = item['sources']

                    st.markdown("---")
                    st.markdown(f"**🔍 Web Search {search_counter}:** {query} ({len(sources)} sources)")

                    for idx, source in enumerate(sources, 1):
                        st.markdown(f"{idx}. [{source['url']}]({source['url']})")

                    st.markdown("")

    if issues:
        st.markdown("### Highlighted Text")

        # Always show all categories
        highlighted_text = highlight_text(current_text, issues, True, True, True)

        # HTML content with highlighting and tooltips
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        body {{
            font-family: "Source Sans Pro", sans-serif;
            font-size: 16px;
            line-height: 1.6;
            color: white;
            margin: 0;
            padding: 10px;
            padding-bottom: 300px;
            overflow: visible;
        }}
        .fact-issue {{
            position: relative;
            transition: all 0.2s ease;
        }}
        .fact-issue:hover {{
            border-width: 3px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .fact-issue::after {{
            content: attr(data-badge-text);
            position: absolute;
            top: -8px;
            right: -8px;
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 11px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid white;
            z-index: 100;
        }}
        .fact-issue[data-badge-type="single"]::after {{
            background: #666;
        }}
        .fact-issue[data-badge-type="multiple"]::after {{
            background: #ff6600;
        }}
        .custom-tooltip {{
            position: absolute;
            background: #333;
            color: white;
            padding: 12px 28px 12px 12px;
            border-radius: 6px;
            z-index: 10000;
            max-width: 700px;
            min-width: 400px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .custom-tooltip a {{
            color: #66b3ff;
            text-decoration: underline;
        }}
        .tooltip-close {{
            position: absolute;
            top: 6px;
            right: 8px;
            background: transparent;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            padding: 0;
            width: 20px;
            height: 20px;
            line-height: 20px;
            text-align: center;
        }}
        .tooltip-close:hover {{
            color: #ff6666;
        }}
        </style>
        </head>
        <body>
            {highlighted_text}
            <script>
            let currentTooltip = null;

            function linkifyText(text) {{
                const urlRegex = /(https?:\\/\\/[^\\s<]+)/g;
                return text.replace(urlRegex, '<a href="$1" target="_blank" style="color: #66b3ff; text-decoration: underline;">$1</a>');
            }}

            function resizeIframe() {{
                const height = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight) + 20;
                window.parent.postMessage({{
                    type: 'streamlit:setFrameHeight',
                    height: height
                }}, '*');
            }}

            document.querySelectorAll('.fact-issue').forEach(function(mark) {{
                mark.addEventListener('click', function(e) {{
                    e.stopPropagation();

                    if (currentTooltip) {{
                        currentTooltip.remove();
                        currentTooltip = null;
                        document.body.style.paddingBottom = '300px';
                    }}

                    const tooltipText = this.getAttribute('data-tooltip');
                    if (!tooltipText) return;

                    const tooltip = document.createElement('div');
                    tooltip.className = 'custom-tooltip';

                    let htmlContent = tooltipText.replace(/\\n/g, '<br>');
                    htmlContent = linkifyText(htmlContent);
                    tooltip.innerHTML = htmlContent;

                    const closeBtn = document.createElement('button');
                    closeBtn.className = 'tooltip-close';
                    closeBtn.innerHTML = '&times;';
                    closeBtn.onclick = function(e) {{
                        e.stopPropagation();
                        tooltip.remove();
                        currentTooltip = null;
                        document.body.style.paddingBottom = '300px';
                        setTimeout(resizeIframe, 10);
                    }};
                    tooltip.appendChild(closeBtn);

                    document.body.appendChild(tooltip);
                    const rect = this.getBoundingClientRect();
                    const tooltipRect = tooltip.getBoundingClientRect();
                    const viewportWidth = window.innerWidth || document.documentElement.clientWidth;
                    const viewportHeight = window.innerHeight || document.documentElement.clientHeight;

                    let left = rect.left;
                    let top = rect.bottom + 5;

                    if (left + tooltipRect.width > viewportWidth) {{
                        left = viewportWidth - tooltipRect.width - 10;
                    }}

                    if (left < 10) {{
                        left = 10;
                    }}

                    if (top + tooltipRect.height > viewportHeight) {{
                        top = rect.top - tooltipRect.height - 5;
                    }}

                    if (top < 10) {{
                        top = rect.bottom + 5;
                    }}

                    tooltip.style.left = left + 'px';
                    tooltip.style.top = top + 'px';

                    currentTooltip = tooltip;

                    setTimeout(function() {{
                        const tooltipHeight = tooltip.offsetHeight;
                        document.body.style.paddingBottom = (tooltipHeight + 150) + 'px';
                        setTimeout(resizeIframe, 10);
                    }}, 10);
                }});
            }});

            document.addEventListener('click', function(e) {{
                if (currentTooltip && !currentTooltip.contains(e.target)) {{
                    currentTooltip.remove();
                    currentTooltip = null;
                    document.body.style.paddingBottom = '300px';
                    setTimeout(resizeIframe, 10);
                }}
            }});

            window.addEventListener('load', resizeIframe);
            setTimeout(resizeIframe, 100);
            setTimeout(resizeIframe, 500);
            </script>
        </body>
        </html>
        """

        estimated_lines = len(current_text) / 80
        initial_height = max(400, min(int(estimated_lines * 24) + 400, 3000))

        components.html(html_content, height=initial_height, scrolling=False)

        # Legend
        st.markdown("""
        <div style="margin-top: 20px; margin-bottom: 30px; padding: 10px; background-color: #1e1e1e; border-radius: 5px; color: #e0e0e0; border: 1px solid #3a3a3a;">
            <b>Legend:</b><br>
            <mark style="background-color: #ffcccc; padding: 2px 4px; color: #000;">Misleading</mark>
            <mark style="background-color: #fff4cc; padding: 2px 4px; color: #000;">Questionable</mark>
            <mark style="background-color: #cce5ff; padding: 2px 4px; color: #000;">Incomplete</mark>
            <br><br>
            <b>Issue badges:</b> Each highlight shows a number badge
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li><span style="background: #666; color: white; border-radius: 50%; padding: 2px 6px; font-size: 11px; font-weight: bold;">3</span> = Issue #3</li>
                <li><span style="background: #ff6600; color: white; border-radius: 50%; padding: 2px 6px; font-size: 11px; font-weight: bold;">2</span> = Multiple overlapping issues</li>
            </ul>
            <small>💡 Click on highlighted text to see full explanations and sources. Click the × or outside the tooltip to close.</small>
        </div>
        """, unsafe_allow_html=True)

        # Issues section
        st.markdown("### Issues Found")

        type_order = {'misleading': 0, 'incomplete': 1, 'questionable': 2}
        sorted_issues = sorted(enumerate(issues), key=lambda x: type_order.get(x[1].get('type', 'questionable'), 3))

        for original_index, issue in sorted_issues:
            issue_type = issue.get('type', 'questionable').title()
            issue_type_lower = issue.get('type', 'questionable')

            excerpt = issue.get('excerpt', 'N/A')
            explanation = issue.get('issue', 'No explanation provided')
            issue_sources = issue.get('sources', [])

            type_colors = {
                'Misleading': '🔴',
                'Questionable': '🟡',
                'Incomplete': '🔵'
            }
            icon = type_colors.get(issue_type, '⚪')

            # Border colors for each type
            border_colors = {
                'Misleading': '#ff4444',
                'Questionable': '#ffaa00',
                'Incomplete': '#4499ff'
            }
            border_color = border_colors.get(issue_type, '#555')

            sources_html = ""
            if issue_sources:
                sources_html = "<br><br><b>Sources:</b><br>"
                for idx, src in enumerate(issue_sources, 1):
                    if src.startswith('http'):
                        sources_html += f'{idx}. <a href="{src}" target="_blank" style="color: #66b3ff;">{src}</a><br>'
                    else:
                        sources_html += f'{idx}. {src}<br>'

            st.markdown(f"""
            <div id="issue-{original_index}" style="padding: 10px; margin-bottom: 15px; border-left: 4px solid {border_color}; background-color: #2b2b2b; color: #e0e0e0;">
                <b>{icon} Issue #{original_index+1}: {issue_type}</b><br>
                <i style="color: #999;">"{excerpt[:100]}{'...' if len(excerpt) > 100 else ''}"</i><br><br>
                <div>{explanation}</div>
                {sources_html}
            </div>
            """, unsafe_allow_html=True)

        # Display all sources
        if all_sources:
            st.markdown("---")
            st.markdown("### 📚 All Sources Used")
            for idx, source in enumerate(all_sources, 1):
                if source.startswith('http'):
                    st.markdown(f"{idx}. [{source}]({source})")
                else:
                    st.markdown(f"{idx}. {source}")
    else:
        st.success("✅ No issues found! The text appears to be accurate and complete.")

        if all_sources:
            st.markdown("---")
            st.markdown("### 📚 Sources Consulted")
            for idx, source in enumerate(all_sources, 1):
                if source.startswith('http'):
                    st.markdown(f"{idx}. [{source}]({source})")
                else:
                    st.markdown(f"{idx}. {source}")

    st.caption(f"Cost: ${results['cost']:.3f} · {results['search_count']} web searches · {results['duration']:.0f}s")
