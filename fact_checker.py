import streamlit as st
import streamlit.components.v1 as components
import os
import json
import re
import traceback
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenRouter client (OpenAI-compatible)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

st.set_page_config(page_title="Fact Checker", page_icon="🔍", layout="wide")

st.title("🔍 Fact Checker")
st.write("Enter text below to fact-check it for misleading, questionable, or incomplete information.")

# Initialize session state
if 'main_text_input' not in st.session_state:
    st.session_state.main_text_input = ""

if 'fact_check_results' not in st.session_state:
    st.session_state.fact_check_results = None

# Hardcoded model configuration
MODEL_CHOICE = "openai/gpt-5"
REASONING_EFFORT = "high"
ENABLE_STREAMING = True

# Paste helper to extract links from Google Docs
paste_helper = """
<script>
function parseHtmlWithLinks(html) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');

    function processNode(node) {
        if (node.nodeType === Node.TEXT_NODE) {
            return node.textContent;
        }

        if (node.nodeType === Node.ELEMENT_NODE) {
            if (node.tagName === 'A' && node.href) {
                return `${node.textContent} (${node.href})`;
            }

            if (node.tagName === 'BR') {
                return '\\n';
            }

            if (['P', 'DIV', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'LI'].includes(node.tagName)) {
                let text = '';
                for (let child of node.childNodes) {
                    text += processNode(child);
                }
                return text + '\\n';
            }

            let text = '';
            for (let child of node.childNodes) {
                text += processNode(child);
            }
            return text;
        }
        return '';
    }

    let result = processNode(doc.body);
    return result.replace(/\\n{3,}/g, '\\n\\n').trim();
}

function attachPasteHandler() {
    const textareas = window.parent.document.querySelectorAll('textarea');

    textareas.forEach(textarea => {
        if (textarea.dataset.pasteHandlerAttached) return;
        textarea.dataset.pasteHandlerAttached = 'true';

        textarea.addEventListener('paste', function(e) {
            const clipboardData = e.clipboardData || window.clipboardData;
            const htmlData = clipboardData.getData('text/html');

            if (htmlData && htmlData.includes('<a')) {
                e.preventDefault();

                const processedText = parseHtmlWithLinks(htmlData);
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
    placeholder="Paste your text here (links from Google Docs will be preserved automatically)...",
    key="main_text_input"
)

submit_button = st.button("Fact Check", type="primary")


def build_position_map(text):
    """Build mapping from stripped text (no URLs) to original text positions"""
    stripped = []
    position_map = []
    i = 0

    while i < len(text):
        url_match = re.match(r'\s*\(https?://[^\)]+\)', text[i:])
        if url_match:
            i += len(url_match.group())
            continue

        stripped.append(text[i])
        position_map.append(i)
        i += 1

    return ''.join(stripped), position_map


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

    original_stripped, pos_map = build_position_map(original_text)

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

        start_pos = None
        end_pos = None

        # Try exact match
        pos = original_text.find(excerpt)
        if pos != -1:
            start_pos = pos
            end_pos = pos + len(excerpt)
        else:
            # Try stripped text match
            excerpt_stripped, _ = build_position_map(excerpt)
            pos = original_stripped.find(excerpt_stripped)

            if pos != -1 and len(excerpt_stripped) > 0:
                start_pos = pos_map[pos]
                end_pos_stripped = min(pos + len(excerpt_stripped), len(pos_map))

                if end_pos_stripped > 0 and end_pos_stripped <= len(pos_map):
                    end_pos = pos_map[end_pos_stripped - 1] + 1

        if start_pos is not None and end_pos is not None:
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
        with st.spinner("Performing deep research and fact-checking..."):
            try:
                prompt = f"""
Run a deep research to fact check this text. Identify any misleading information, questionable statements, or missing important information that would confuse the reader.

CRITICAL INSTRUCTIONS:
1. Respond in English
2. For the "excerpt" field: You MUST copy-paste the EXACT text from the original that has the issue. DO NOT write your own summary or commentary.
3. If information is MISSING (incomplete), you can describe what's missing in the "issue" field, but leave "excerpt" empty or put a relevant sentence that should have more detail.
4. Only include issues where you can point to specific problematic text OR identify specific gaps.

Return your analysis as a JSON object with the following structure:
{{
"issues": [
    {{
    "excerpt": "EXACT TEXT copied from the original (not your summary)",
    "issue": "explanation of what is wrong, misleading, or missing",
    "type": "misleading" | "questionable" | "incomplete",
    "sources": ["URL or source 1", "URL or source 2"]
    }}
],
"all_sources": ["list", "of", "all", "sources", "used"]
}}

For each issue, provide the sources you used to verify the information. Include ALL sources at the end in the all_sources array.

If no issues are found, return: {{"issues": [], "all_sources": []}}

Text to fact-check:
{current_text}
"""

                json_schema = {
                    "type": "object",
                    "properties": {
                        "issues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "excerpt": {"type": "string"},
                                    "issue": {"type": "string"},
                                    "type": {"type": "string", "enum": ["misleading", "questionable", "incomplete"]},
                                    "sources": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["excerpt", "issue", "type", "sources"],
                                "additionalProperties": False
                            }
                        },
                        "all_sources": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["issues", "all_sources"],
                    "additionalProperties": False
                }

                status_placeholder = st.empty()
                streaming_placeholder = st.empty()

                api_params = {
                    "model": MODEL_CHOICE,
                    "messages": [
                        {"role": "system", "content": "You are a professional fact-checker. Analyze texts thoroughly and identify misleading information, questionable statements, and missing context. Always respond in English."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": ENABLE_STREAMING,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "fact_check_results",
                            "strict": True,
                            "schema": json_schema
                        }
                    }
                }

                extra_body = {}
                if REASONING_EFFORT:
                    extra_body["reasoning"] = {
                        "effort": REASONING_EFFORT,
                        "summary": "auto"
                    }
                if extra_body:
                    api_params["extra_body"] = extra_body

                response = client.chat.completions.create(**api_params)

                result_text = ""
                reasoning_text = ""
                reasoning_and_search_items = []

                if ENABLE_STREAMING:
                    for chunk in response:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'reasoning_details') and delta.reasoning_details:
                                detail = delta.reasoning_details
                                if isinstance(detail, str):
                                    reasoning_text += detail
                                elif isinstance(detail, list):
                                    for item in detail:
                                        if isinstance(item, dict) and 'text' in item:
                                            reasoning_text += item['text']
                                        elif hasattr(item, 'text'):
                                            reasoning_text += item.text
                            if delta and delta.content:
                                result_text += delta.content
                                streaming_placeholder.markdown(f"**Live Response:**\n```json\n{result_text}\n```")

                    if reasoning_text:
                        reasoning_and_search_items.append({
                            'type': 'reasoning',
                            'text': reasoning_text
                        })

                    status_placeholder.empty()
                    streaming_placeholder.empty()
                else:
                    if response.choices and len(response.choices) > 0:
                        msg = response.choices[0].message
                        result_text = msg.content or ""
                        reasoning_content = getattr(msg, 'reasoning', None)
                        if reasoning_content:
                            reasoning_and_search_items.append({
                                'type': 'reasoning',
                                'text': reasoning_content
                            })

                # Parse results
                if not result_text or result_text.strip() == "":
                    st.error("No response received from the API. Please check your API key and model availability.")
                    issues = []
                    all_sources = []
                else:
                    result_json = json.loads(result_text)
                    if isinstance(result_json, dict):
                        issues = result_json.get('issues', [])
                        all_sources = result_json.get('all_sources', [])
                    else:
                        issues = result_json
                        all_sources = []

                st.success("Fact-check complete!")

                st.session_state.fact_check_results = {
                    'issues': issues,
                    'all_sources': all_sources,
                    'current_text': current_text,
                    'model_choice': MODEL_CHOICE,
                    'reasoning_and_search_items': reasoning_and_search_items
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
    model_choice = results['model_choice']
    reasoning_and_search_items = results['reasoning_and_search_items']

    # Display reasoning summary and web searches
    if "gpt-5" in model_choice and reasoning_and_search_items:
        with st.expander("🧠 Reasoning Summary & Web Search", expanded=True):
            search_counter = 0
            for item in reasoning_and_search_items:
                if item['type'] == 'reasoning':
                    st.markdown(item['text'])
                    st.markdown("")

                elif item['type'] == 'web_search':
                    search_counter += 1
                    query = item.get('query', '')
                    sources = item.get('sources', [])

                    st.markdown("---")
                    if query:
                        st.markdown(f"**🔍 Web Search {search_counter}:** {query}")
                    else:
                        st.markdown(f"**🔍 Web Search {search_counter}**")

                    if sources:
                        for idx, source in enumerate(sources, 1):
                            url = source.get('url', '') if isinstance(source, dict) else (source.url if hasattr(source, 'url') else str(source))
                            if url:
                                st.markdown(f"{idx}. [{url}]({url})")

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


