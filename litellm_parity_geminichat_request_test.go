package protocol

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// This file pins the Chat<->Gemini request folds against LiteLLM v1.100.1.
//
// Truth source for the chat->gemini direction (the reference has a forward
// path there): every expected body below was produced by running the
// reference adapters from the repo venv:
//
//	python3
//	  litellm.get_optional_params(model="gemini-2.5-pro",
//	      custom_llm_provider="vertex_ai", drop_params=<drop>, <params>)
//	  -> litellm/llms/vertex_ai/gemini/transformation.py:sync_transform_request_body
//	     (message assembly in _gemini_convert_messages_with_history,
//	     transformation.py:723-1069; body assembly transformation.py:1217-1247)
//
// The gemini->chat direction has no forward path in the reference: its
// expectations are derived from the spec's inverse rules (each row cites the
// forward mapping it inverts) and were validated by feeding this codec's
// output back through the reference's forward adapters (round-trip check run
// during this audit; the lossy cases are asserted as documented drops).

// geminiChatCanonical renders a decoded JSON value so two bodies compare
// independent of key order (Go maps marshal with sorted keys).
func geminiChatCanonical(t *testing.T, v string) string {
	t.Helper()
	var parsed any
	if err := json.Unmarshal([]byte(v), &parsed); err != nil {
		t.Fatalf("bad JSON %q: %v", v, err)
	}
	b, err := json.Marshal(parsed)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func geminiChatConvertRequest(t *testing.T, body string, drop bool) string {
	t.Helper()
	out, err := ConvertRequestWith(Chat, Gemini, []byte(body), ConvertOptions{DropParams: drop})
	if err != nil {
		t.Fatalf("%s: %v", body, err)
	}
	return string(out.Body)
}

// TestGeminiChatParityChatToGeminiRequest pins the request fold against
// reference-ran truth. Adapter: sync_transform_request_body
// (litellm/llms/vertex_ai/gemini/transformation.py:299) fed by
// get_optional_params over VertexGeminiConfig.map_openai_params
// (litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:296-327).
func TestGeminiChatParityChatToGeminiRequest(t *testing.T) {
	cases := []struct{ name, chat, want string }{
		// Minimal: user string -> one text part (transformation.py:728-731, 850-852).
		{"minimal", `{"messages":[{"role":"user","content":"hello"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}`},
		// System + sampling + tools + tool_choice=required
		// (system extraction transformation.py:1399-1412; sampling
		// vertex_and_google_ai_studio_gemini.py:1090-1121; tools
		// vertex_and_google_ai_studio_gemini.py:660-703; tool_choice
		// vertex_and_google_ai_studio_gemini.py:331-346).
		{"system params tools", `{"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],"temperature":0.5,"top_p":0.9,"max_tokens":32,"stop":"END","tools":[{"type":"function","function":{"name":"lookup","description":"d","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}],"tool_choice":"required"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"system_instruction":{"parts":[{"text":"be brief"}]},"tools":[{"function_declarations":[{"name":"lookup","description":"d","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}]}],"toolConfig":{"functionCallingConfig":{"mode":"ANY"}},"generationConfig":{"temperature":0.5,"top_p":0.9,"max_output_tokens":32,"stop_sequences":["END"]}}`},
		// Tool history: assistant tool_calls -> function_call (args parsed,
		// no id written - 2.5 semantics, factory.py:1123-1138 and
		// vertex_and_google_ai_studio_gemini.py:278-285), tool result ->
		// function_response under {"content": ...} (factory.py:1465-1500,
		// name resolved from the call by tool_call_id factory.py:1443-1449).
		{"tool history", `{"messages":[{"role":"user","content":"weather?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Boston\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"sunny"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"weather?"}]},{"role":"model","parts":[{"function_call":{"name":"get_weather","args":{"city":"Boston"}},"thoughtSignature":"skip_thought_signature_validator"}]},{"role":"user","parts":[{"function_response":{"name":"get_weather","response":{"content":"sunny"}}}]}]}`},
		// Tool result whose text is a JSON object rides as the response itself
		// (factory.py:1465-1476); empty arguments string -> placeholder object
		// (factory.py:1123-1128); a list result contributes its text plus inline
		// media inside the function response (transformation.py:725-849).
		{"tool result json", `{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"{\"a\": 1}"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"x"}]},{"role":"model","parts":[{"function_call":{"args":{},"name":"f"},"thoughtSignature":"skip_thought_signature_validator"}]},{"role":"user","parts":[{"function_response":{"name":"f","response":{"a":1}}}]}]}`},
		{"tool result media list", `{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":""}}]},{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"see"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAB"}}]}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"x"}]},{"role":"model","parts":[{"function_call":{"args":{"type":"object"},"name":"f"},"thoughtSignature":"skip_thought_signature_validator"}]},{"role":"user","parts":[{"function_response":{"name":"f","parts":[{"inline_data":{"data":"AAAB","mime_type":"image/png"}}],"response":{"content":"see"}}}]}]}`},
		// reasoning_effort=high + json_schema with strict stripped recursively
		// (vertex_and_google_ai_studio_gemini.py:768-790, 1155-1177;
		// litellm/utils.py:3790-3808).
		{"effort schema", `{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high","response_format":{"type":"json_schema","json_schema":{"name":"out","strict":true,"schema":{"type":"object","properties":{"a":{"type":"string"}}}}}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":4096,"includeThoughts":true},"response_mime_type":"application/json","response_json_schema":{"type":"object","properties":{"a":{"type":"string"}}}}}`},
		// The thinking object spelling (vertex_and_google_ai_studio_gemini.py:940-970).
		{"thinking param", `{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":2048}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"includeThoughts":true,"thinkingBudget":2048}}}`},
		// Image data URL -> inline_data (transformation.py:575-579); multiturn.
		{"image multiturn", `{"messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAB"}}]},{"role":"assistant","content":"I see"},{"role":"user","content":"and?"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"look"},{"inline_data":{"data":"AAAB","mime_type":"image/png"}}]},{"role":"model","parts":[{"text":"I see"}]},{"role":"user","parts":[{"text":"and?"}]}]}`},
		// System-only request gains the "." user turn
		// (transformation.py:1375-1381, 1421-1423).
		{"system only", `{"messages":[{"role":"system","content":"be brief"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"."}]}],"system_instruction":{"parts":[{"text":"be brief"}]}}`},
		// Consecutive same-role turns merge into one content
		// (transformation.py:723-724 user group, 873 assistant group).
		{"merge same role", `{"messages":[{"role":"user","content":"a"},{"role":"user","content":"b"},{"role":"assistant","content":"c"},{"role":"assistant","content":"d"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"a"},{"text":"b"}]},{"role":"model","parts":[{"text":"c"},{"text":"d"}]}]}`},
		// A system turn mid-conversation is lifted to system_instruction and
		// its slot vanishes from contents (transformation.py:701, 1399-1412).
		{"system midstream", `{"messages":[{"role":"user","content":"a"},{"role":"system","content":"sys!"},{"role":"user","content":"b"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"a"},{"text":"b"}]}],"system_instruction":{"parts":[{"text":"sys!"}]}}`},
		// System list content joins its texts with no separator
		// (transformation.py:1399-1412).
		{"system list content", `{"messages":[{"role":"system","content":[{"type":"text","text":"one"},{"type":"text","text":"two"}]},{"role":"user","content":"q"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"q"}]}],"system_instruction":{"parts":[{"text":"onetwo"}]}}`},
		// Each system message is its own instruction part (transformation.py:1399-1412).
		{"two systems", `{"messages":[{"role":"system","content":"s1"},{"role":"system","content":"s2"},{"role":"user","content":"q"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"q"}]}],"system_instruction":{"parts":[{"text":"s1"},{"text":"s2"}]}}`},
		// The blank-text cure: no text part anywhere in a user content appends
		// {"text":" "} at the end (common_utils.py:494-505, transformation.py:856-870).
		{"blank text empty list", `{"messages":[{"role":"user","content":[]}]}`,
			`{"contents":[{"role":"user","parts":[{"text":" "}]}]}`},
		{"blank text empty text part", `{"messages":[{"role":"user","content":[{"type":"text","text":""}]}]}`,
			`{"contents":[{"role":"user","parts":[{"text":" "}]}]}`},
		{"blank text null content", `{"messages":[{"role":"user","content":null}]}`,
			`{"contents":[{"role":"user","parts":[{"text":" "}]}]}`},
		{"image only blank text", `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}]}]}`,
			`{"contents":[{"role":"user","parts":[{"inline_data":{"data":"AA","mime_type":"image/png"}},{"text":" "}]}]}`},
		// A user string rides verbatim however empty
		// (transformation.py:850-852 has no length gate).
		{"user empty string", `{"messages":[{"role":"user","content":""}]}`,
			`{"contents":[{"role":"user","parts":[{"text":""}]}]}`},
		// Empty messages list -> the default user turn
		// (transformation.py:1069).
		{"empty messages", `{"messages":[]}`,
			`{"contents":[{"role":"user","parts":[{"text":" "}]}]}`},
		// reasoning_content -> thought part (transformation.py:880-883).
		{"assistant reasoning", `{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","reasoning_content":"why"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"why","thought":true},{"text":"a"}]}]}`},
		// An empty or blank reasoning summary mints no part — the same
		// truthiness gate the reference applies, and the empty thought part
		// it would produce is a 400 on strict upstreams.
		{"assistant empty reasoning", `{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","reasoning_content":""}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"a"}]}]}`},
		{"assistant blank reasoning", `{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","reasoning_content":"  "}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"a"}]}]}`},
		// Signed thinking block -> a thought part carrying the signature
		// itself (deliberate deviation from transformation.py:884-901, whose
		// plain-text signature carrier loses the signature on the way back —
		// the chat reverse reads signatures only off thought parts); unsigned
		// and redacted blocks still drop.
		{"assistant thinking blocks", `{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","thinking_blocks":[{"type":"thinking","thinking":"signed","signature":"SIG"},{"type":"thinking","thinking":"unsigned"},{"type":"redacted_thinking","data":"xx"}]}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"signed","thought":true,"thoughtSignature":"SIG"},{"text":"a"}]}]}`},
		// Assistant list content keeps only text parts (transformation.py:904-912).
		{"assistant list text only", `{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":[{"type":"text","text":"t1"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}},{"type":"text","text":"t2"}]}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"t1"},{"text":"t2"}]}]}`},
		// The assistant images field rides inline media
		// (transformation.py:943-966).
		{"assistant images field", `{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAB"},"index":0}]}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"a"},{"inline_data":{"data":"AAAB","mime_type":"image/png"}}]}]}`},
		// stop variants (vertex_and_google_ai_studio_gemini.py:1115-1119).
		{"stop list", `{"messages":[{"role":"user","content":"hi"}],"stop":["a","b"]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"stop_sequences":["a","b"]}}`},
		{"stop empty list", `{"messages":[{"role":"user","content":"hi"}],"stop":[]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"stop_sequences":[]}}`},
		{"stop empty string", `{"messages":[{"role":"user","content":"hi"}],"stop":""}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"stop_sequences":[""]}}`},
		// n=1 still states candidate_count
		// (vertex_and_google_ai_studio_gemini.py:1111-1112).
		{"n one", `{"messages":[{"role":"user","content":"hi"}],"n":1}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"candidate_count":1}}`},
		// max_completion_tokens wins (vertex_and_google_ai_studio_gemini.py:1120-1121).
		{"max completion tokens", `{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":17}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"max_output_tokens":17}}`},
		// response_format variants (vertex_and_google_ai_studio_gemini.py:768-775).
		{"rf json_object", `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"response_mime_type":"application/json"}}`},
		{"rf text", `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"text"}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"response_mime_type":"text/plain"}}`},
		{"rf schema without schema", `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"o"}}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`},
		// response_json_schema keeps additionalProperties:false (the strict
		// removal is recursive but the properties stay; utils.py:3790-3808).
		{"rf schema additionalProperties", `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"o","strict":true,"schema":{"type":"object","additionalProperties":false,"properties":{"a":{"type":"string"}}}}}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"response_mime_type":"application/json","response_json_schema":{"type":"object","additionalProperties":false,"properties":{"a":{"type":"string"}}}}}`},
		// Tool schema pipeline: strict and false additionalProperties stripped
		// over the whole list (utils.py:3767-3808), ["string","null"] ->
		// anyOf (common_utils.py:892-935).
		{"tool schema type array", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","strict":true,"function":{"name":"f","strict":true,"parameters":{"type":"object","additionalProperties":false,"properties":{"a":{"type":["string","null"]}},"required":["a"]}}}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f","parameters":{"type":"object","properties":{"a":{"anyOf":[{"type":"string"},{"type":"null"}]}},"required":["a"]}}]}]}`},
		// $defs inlined (common_utils.py:593-598).
		{"tool schema defs", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{"$ref":"#/$defs/A"}},"$defs":{"A":{"type":"string"}}}}}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f","parameters":{"type":"object","properties":{"a":{"type":"string"}}}}]}]}`},
		// Empty enum string -> null (common_utils.py:509-526).
		{"tool schema enum empty string", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{"type":"string","enum":["","x"]}}}}}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f","parameters":{"type":"object","properties":{"a":{"type":"string","enum":[null,"x"]}}}}]}]}`},
		// Empty properties collapse to bare {"type":"object"}
		// (common_utils.py:815-830).
		{"tool empty properties", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f","parameters":{"type":"object"}}]}]}`},
		// No parameters -> no parameters key; empty-string parameters ->
		// {"type":"object"} (vertex_and_google_ai_studio_gemini.py:655-674).
		{"tool no parameters", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f"}]}]}`},
		{"tool parameters empty string", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":""}}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f","parameters":{"type":"object"}}]}]}`},
		// Several functions merge into one Tools entry
		// (vertex_and_google_ai_studio_gemini.py:699-703).
		{"multiple functions merge", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f1","parameters":{"type":"object"}}},{"type":"function","function":{"name":"f2","parameters":{"type":"object"}}}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f1","parameters":{"type":"object"}},{"name":"f2","parameters":{"type":"object"}}]}]}`},
		// web_search_options -> one googleSearch tool entry
		// (vertex_and_google_ai_studio_gemini.py:348-354).
		{"web search", `{"messages":[{"role":"user","content":"hi"}],"web_search_options":{}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"googleSearch":{}}]}`},
		// Search tools mixed with function declarations drop the search
		// (vertex_and_google_ai_studio_gemini.py:370-406, 516-563).
		{"search mixed with functions", `{"messages":[{"role":"user","content":"hi"}],"web_search_options":{},"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f","parameters":{"type":"object"}}]}]}`},
		// tool_choice modes (vertex_and_google_ai_studio_gemini.py:331-346);
		// a dict without a function name yields [""] (reference-ran).
		{"tool_choice none", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"none"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f","parameters":{"type":"object"}}]}],"toolConfig":{"functionCallingConfig":{"mode":"NONE"}}}`},
		{"tool_choice auto", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"auto"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f","parameters":{"type":"object"}}]}],"toolConfig":{"functionCallingConfig":{"mode":"AUTO"}}}`},
		{"tool_choice named", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"lookup"}}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"lookup","parameters":{"type":"object"}}]}],"toolConfig":{"functionCallingConfig":{"mode":"ANY","allowed_function_names":["lookup"]}}}`},
		{"tool_choice named missing name", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{}}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f","parameters":{"type":"object"}}]}],"toolConfig":{"functionCallingConfig":{"mode":"ANY","allowed_function_names":[""]}}}`},
		// safety_settings / cached_content ride verbatim
		// (transformation.py:1233, 1237).
		{"safety settings", `{"messages":[{"role":"user","content":"hi"}],"safety_settings":[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"safetySettings":[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"}]}`},
		{"cached content", `{"messages":[{"role":"user","content":"hi"}],"cached_content":"caches/1"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"cachedContent":"caches/1"}`},
		// service_tier (transformation.py:1239-1246,
		// vertex_and_google_ai_studio_gemini.py:408-417).
		{"service tier auto", `{"messages":[{"role":"user","content":"hi"}],"service_tier":"auto"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"serviceTier":"priority"}`},
		{"service tier default", `{"messages":[{"role":"user","content":"hi"}],"service_tier":"default"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"serviceTier":"standard"}`},
		{"service tier other", `{"messages":[{"role":"user","content":"hi"}],"service_tier":"batch"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"serviceTier":"batch"}`},
		// effort table (vertex_and_google_ai_studio_gemini.py:799-846,
		// 1155-1177).
		{"effort minimal", `{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"minimal"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":128,"includeThoughts":true}}}`},
		{"effort low", `{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"low"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":1024,"includeThoughts":true}}}`},
		{"effort medium", `{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"medium"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":2048,"includeThoughts":true}}}`},
		{"effort high", `{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":4096,"includeThoughts":true}}}`},
		{"effort none", `{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":0,"includeThoughts":false}}}`},
		{"effort disable", `{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"disable"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":0,"includeThoughts":false}}}`},
		// thinking object variants (vertex_and_google_ai_studio_gemini.py:940-970):
		// enabled+0 budget -> {"thinkingBudget":0}; disabled/adaptive -> the
		// empty config, still sent.
		{"thinking enabled zero budget", `{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":0}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":0}}}`},
		{"thinking disabled", `{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{}}}`},
		{"thinking adaptive", `{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"adaptive"}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{}}}`},
		{"thinking enabled no budget", `{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled"}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"includeThoughts":true}}}`},
		// An explicit thinking object overrides reasoning_effort
		// (vertex_and_google_ai_studio_gemini.py:1155 then 1179).
		{"thinking overrides effort", `{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"low","thinking":{"type":"enabled","budget_tokens":512}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"includeThoughts":true,"thinkingBudget":512}}}`},
		// modalities: a list maps member-wise; a string is dropped (the gate
		// is isinstance(value, list),
		// vertex_and_google_ai_studio_gemini.py:1190); an empty list rides.
		{"modalities list", `{"messages":[{"role":"user","content":"hi"}],"modalities":["text","image"]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"responseModalities":["TEXT","IMAGE"]}}`},
		{"modalities string dropped", `{"messages":[{"role":"user","content":"hi"}],"modalities":"text"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`},
		{"modalities empty list", `{"messages":[{"role":"user","content":"hi"}],"modalities":[]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"responseModalities":[]}}`},
		// audio builds speechConfig and appends the AUDIO modality
		// (vertex_and_google_ai_studio_gemini.py:1000-1046, 1204-1209).
		{"audio", `{"messages":[{"role":"user","content":"hi"}],"audio":{"voice":"Charon","format":"pcm16"}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Charon"}}}}}`},
		{"audio with modalities", `{"messages":[{"role":"user","content":"hi"}],"audio":{"voice":"Charon","format":"pcm16"},"modalities":["text"]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"responseModalities":["TEXT","AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Charon"}}}}}`},
		{"audio language", `{"messages":[{"role":"user","content":"hi"}],"audio":{"language_code":"en-US"}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"languageCode":"en-US"}}}`},
		// logprobs pair (vertex_and_google_ai_studio_gemini.py:1131-1133).
		{"logprobs", `{"messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":2}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"responseLogprobs":true,"logprobs":2}}`},
		{"penalties seed topk", `{"messages":[{"role":"user","content":"hi"}],"presence_penalty":0.1,"frequency_penalty":0.2,"seed":7,"top_k":3}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"presence_penalty":0.1,"frequency_penalty":0.2,"seed":7,"top_k":3}}`},
		// stream:true rides the params, not the body (transformation.py:1203).
		{"stream true no body change", `{"messages":[{"role":"user","content":"hi"}],"stream":true}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`},
		// parallel_tool_calls has no wire field either way
		// (vertex_and_google_ai_studio_gemini.py:1145-1153).
		{"ptc false single tool", `{"messages":[{"role":"user","content":"hi"}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f","parameters":{"type":"object"}}]}]}`},
		{"ptc false multi tools", `{"messages":[{"role":"user","content":"hi"}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"f1","parameters":{"type":"object"}}},{"type":"function","function":{"name":"f2","parameters":{"type":"object"}}}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"function_declarations":[{"name":"f1","parameters":{"type":"object"}},{"name":"f2","parameters":{"type":"object"}}]}]}`},
		// Unknown keys and the unvalidated-but-unread params ride nothing
		// (transformation.py:1203 field whitelist; litellm/utils.py:4772).
		{"unknown key dropped", `{"messages":[{"role":"user","content":"hi"}],"foo":"bar"}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`},
		{"unvalidated dropped", `{"messages":[{"role":"user","content":"hi"}],"user":"u","stream_options":{"include_usage":true},"timeout":5,"extra_headers":{"a":"b"},"speed":"fast","cache_control":{"type":"ephemeral"}}`,
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`},
		// https URL with a known extension -> file_data
		// (transformation.py:569-573).
		{"https image known ext", `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/pic.jpg"}}]}]}`,
			`{"contents":[{"role":"user","parts":[{"file_data":{"file_uri":"https://example.com/pic.jpg","mime_type":"image/jpeg"}},{"text":" "}]}]}`},
		// The uploaded-files namespace passes through unfetched
		// (transformation.py:559-573).
		{"files api passthrough", `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://generativelanguage.googleapis.com/v1beta/files/abc123?alt=media"}}]}]}`,
			`{"contents":[{"role":"user","parts":[{"file_data":{"file_uri":"https://generativelanguage.googleapis.com/v1beta/files/abc123?alt=media"}},{"text":" "}]}]}`},
		// gs:// object -> file_data with the extension's type
		// (transformation.py:556-559).
		{"gs uri", `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"gs://bucket/img.png"}}]}]}`,
			`{"contents":[{"role":"user","parts":[{"file_data":{"file_uri":"gs://bucket/img.png","mime_type":"image/png"}},{"text":" "}]}]}`},
		// input_audio -> inline_data (transformation.py:725-849 media branch).
		{"input audio", `{"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"QUJD","format":"wav"}}]}]}`,
			`{"contents":[{"role":"user","parts":[{"inline_data":{"data":"QUJD","mime_type":"audio/wav"}},{"text":" "}]}]}`},
		// file part with a data URL -> inline_data (transformation.py:575-579).
		{"file part data url", `{"messages":[{"role":"user","content":[{"type":"file","file":{"file_data":"data:application/pdf;base64,JVBE","filename":"d.pdf"}}]}]}`,
			`{"contents":[{"role":"user","parts":[{"inline_data":{"data":"JVBE","mime_type":"application/pdf"}},{"text":" "}]}]}`},
		// A tool call's thought signature rides the call part
		// (factory.py:1282-1290 via provider_specific_fields).
		{"tool call signature psf", `{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"},"provider_specific_fields":{"thought_signature":"SIG"}}]},{"role":"tool","tool_call_id":"c1","content":"r"}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"function_call":{"args":{},"name":"f"},"thoughtSignature":"SIG"}]},{"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"r"}}}]}]}`},
		// The turn's text signature rides the text part when no call of the
		// same message already carries it (transformation.py:916-941).
		{"text signature psf", `{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","provider_specific_fields":{"thought_signatures":["SIG"]}}]}`,
			`{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"a","thoughtSignature":"SIG"}]}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := geminiChatConvertRequest(t, tc.chat, false)
			if g, w := geminiChatCanonical(t, got), geminiChatCanonical(t, tc.want); g != w {
				t.Fatalf("chat body %s\n got  %s\n want %s", tc.chat, g, w)
			}
		})
	}
}

// The refusal set: the reference raises for each of these when drop_params is
// off. Adapter: litellm.get_optional_params(custom_llm_provider="vertex_ai")
// raising UnsupportedParamsError (litellm/utils.py:4186-4213) or the config's
// own ValueError / Exception; every outcome below was reference-ran.
func TestGeminiChatParityChatToGeminiRefusals(t *testing.T) {
	cases := []struct {
		name  string
		chat  string
		param string // "" -> a ConversionError, not an UnsupportedParamError
		ref   string // what the reference raised
	}{
		{"logit_bias", `{"messages":[{"role":"user","content":"hi"}],"logit_bias":{"1":2}}`, "logit_bias", "UnsupportedParamsError"},
		{"context_management", `{"messages":[{"role":"user","content":"hi"}],"context_management":{"edits":[]}}`, "context_management", "UnsupportedParamsError"},
		{"tool_choice unknown", `{"messages":[{"role":"user","content":"hi"}],"tool_choice":"banana"}`, "tool_choice", "UnsupportedParamsError (vertex_and_google_ai_studio_gemini.py:331-346)"},
		{"reasoning_effort invalid", `{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"xhigh"}`, "", "ValueError: Invalid reasoning effort (vertex_and_google_ai_studio_gemini.py:1155-1177)"},
		{"audio bad format", `{"messages":[{"role":"user","content":"hi"}],"audio":{"format":"mp3"}}`, "", "ValueError: pcm only (vertex_and_google_ai_studio_gemini.py:1000-1046)"},
		{"unknown role", `{"messages":[{"role":"developer","content":"x"}]}`, "", "Exception: Invalid Message passed in (transformation.py:1052-1055)"},
		{"orphan tool result", `{"messages":[{"role":"user","content":"x"},{"role":"tool","tool_call_id":"nope","content":"r"}]}`, "", "Exception: Missing corresponding tool call (factory.py:1451-1459)"},
		{"tool args bad json", `{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"not json"}}]}]}`, "", "Exception: Unable to convert openai tool calls (factory.py:1130)"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ConvertRequest(Chat, Gemini, []byte(tc.chat))
			if err == nil {
				t.Fatalf("%s: want an error (reference raises %s)", tc.chat, tc.ref)
			}
			if tc.param != "" {
				var unsup UnsupportedParamError
				if !errors.As(err, &unsup) || unsup.Param != tc.param {
					t.Fatalf("%s: err = %T(%v), want UnsupportedParamError(%s)", tc.chat, err, err, tc.param)
				}
			} else {
				var conv ConversionError
				if !errors.As(err, &conv) {
					t.Fatalf("%s: err = %T(%v), want ConversionError", tc.chat, err, err)
				}
			}
		})
	}
}

// drop_params turns every refusal into a drop (litellm/utils.py:4205-413:
// the key is popped rather than raised). Reference-ran: the refused set folds
// to the bare contents body.
func TestGeminiChatParityChatToGeminiDropParams(t *testing.T) {
	body := `{"messages":[{"role":"user","content":"hi"}],"logit_bias":{"1":2},"function_call":"auto","metadata":{"a":1},"prediction":{"type":"content"},"store":true,"context_management":{"edits":[]}}`
	got := geminiChatConvertRequest(t, body, true)
	want := `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`
	if g, w := geminiChatCanonical(t, got), geminiChatCanonical(t, want); g != w {
		t.Fatalf("got  %s\n want %s", g, w)
	}
}

// TestGeminiChatParityGeminiToChatRequest pins the reverse request fold.
// The reference has no forward path here, so each expectation is the spec's
// inverse rule (cited per row) and was validated in this audit by feeding the
// codec's chat output back through the reference's forward adapters
// (sync_transform_request_body), which restored the original gemini body for
// every case except the documented drops asserted at the end of this file.
func TestGeminiChatParityGeminiToChatRequest(t *testing.T) {
	cases := []struct{ name, gemini, want string }{
		// Inverse of the one-text-part fold (transformation.py:728-731): a
		// single text part collapses to a bare string content.
		{"minimal", `{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}`,
			`{"messages":[{"role":"user","content":"hello"}]}`},
		// Inverse of system_instruction (transformation.py:1223, 1399-1412):
		// each instruction part becomes one system message before the rest.
		{"system instruction", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"system_instruction":{"parts":[{"text":"be brief"}]}}`,
			`{"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"q"}]}`},
		{"system instruction two parts", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"system_instruction":{"parts":[{"text":"one"},{"text":"two"}]}}`,
			`{"messages":[{"role":"system","content":"one"},{"role":"system","content":"two"},{"role":"user","content":"q"}]}`},
		{"model text", `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"a"}]}]}`,
			`{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}`},
		// Inverse of transformation.py:880-883: a thought part becomes the
		// assistant's reasoning_content, spelled the same way the reference's
		// own response fold spells it
		// (vertex_and_google_ai_studio_gemini.py:1348-1388, 2307-2322).
		{"thought part", `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"why","thought":true},{"text":"a"}]}]}`,
			`{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","reasoning_content":"why","thinking_blocks":[{"type":"thinking","thinking":"why"}]}]}`},
		// A signed thought part additionally carries the signature the
		// forward fold replays (transformation.py:884-901 inverse; the id
		// pairing rules are the spec's function_response ruling).
		{"thought signature part", `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"why","thought":true,"thoughtSignature":"SIG"}]}]}`,
			`{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":null,"reasoning_content":"why","thinking_blocks":[{"type":"thinking","thinking":"why","signature":"SIG"}],"provider_specific_fields":{"thought_signatures":["SIG"]}}]}`},
		// function_call with its own id (the wire's stable ids) keeps it;
		// arguments render with the reference encoder's separators
		// (vertex_and_google_ai_studio_gemini.py:1562-1600 json.dumps).
		{"function call with id", `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"function_call":{"name":"f","args":{"a":1},"id":"native_1"}}]}]}`,
			`{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":null,"tool_calls":[{"id":"native_1","type":"function","function":{"name":"f","arguments":"{\"a\": 1}"}}]}]}`},
		// A function_call without an id (the 2.5 spelling the forward fold
		// emits) gets a positional id so the result can pair with it
		// (factory.py:1142-1157's id minting, deterministic per D10).
		{"function call no id", `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"function_call":{"name":"f","args":{"a":1}}}]}]}`,
			`{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"f","arguments":"{\"a\": 1}"}}]}]}`},
		// A function_response pairs with the call it answers by name and
		// id; the {"content": string} wrapper folds back to the string
		// (factory.py:1465-1476 inverse).
		{"call and response", `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"function_call":{"name":"f","args":{"a":1}}}]},{"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"r"}}}]}]}`,
			`{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"f","arguments":"{\"a\": 1}"}}]},{"role":"tool","tool_call_id":"call_0","content":"r"}]}`},
		// A response that is a JSON object of its own rides as its compact
		// JSON (factory.py:1465-1476 inverse; spec's compact-JSON rule).
		{"response json payload", `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"function_call":{"name":"f","args":{}}}]},{"role":"user","parts":[{"function_response":{"name":"f","response":{"a":1}}}]}]}`,
			`{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_0","content":"{\"a\":1}"}]}`},
		// User inline_data -> image_url data URL (transformation.py:575-579
		// inverse); mixed parts keep list shape.
		{"user inline data", `{"contents":[{"role":"user","parts":[{"text":"look"},{"inline_data":{"data":"AAAB","mime_type":"image/png"}}]}]}`,
			`{"messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAB"}}]}]}`},
		// file_data -> the URI as the image_url (transformation.py:556-573
		// inverse).
		{"file data", `{"contents":[{"role":"user","parts":[{"file_data":{"mime_type":"image/png","file_uri":"https://example.com/p.png"}}]}]}`,
			`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/p.png"}}]}]}`},
		// Model inline_data rides the images field, where the forward fold
		// reads it back (transformation.py:943-966 inverse).
		{"model inline data", `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"a"},{"inline_data":{"data":"AAAB","mime_type":"image/png"}}]}]}`,
			`{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAB","detail":"auto"},"index":0}]}]}`},
		// function_declarations merge into one chat tools list
		// (vertex_and_google_ai_studio_gemini.py:660-674 inverse).
		{"tools declarations", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"tools":[{"function_declarations":[{"name":"f1","description":"d","parameters":{"type":"object","properties":{"q":{"type":"string"}}}},{"name":"f2"}]}]}`,
			`{"messages":[{"role":"user","content":"q"}],"tools":[{"type":"function","function":{"name":"f1","description":"d","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}},{"type":"function","function":{"name":"f2"}}]}`},
		// googleSearch -> web_search_options
		// (vertex_and_google_ai_studio_gemini.py:348-354 inverse).
		{"tools googleSearch", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"tools":[{"googleSearch":{}}]}`,
			`{"messages":[{"role":"user","content":"q"}],"web_search_options":{}}`},
		// toolConfig modes (vertex_and_google_ai_studio_gemini.py:331-346
		// inverse); a single allowed name names that function.
		{"toolconfig none", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"toolConfig":{"functionCallingConfig":{"mode":"NONE"}}}`,
			`{"messages":[{"role":"user","content":"q"}],"tool_choice":"none"}`},
		{"toolconfig auto", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"toolConfig":{"functionCallingConfig":{"mode":"AUTO"}}}`,
			`{"messages":[{"role":"user","content":"q"}],"tool_choice":"auto"}`},
		{"toolconfig any", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"toolConfig":{"functionCallingConfig":{"mode":"ANY"}}}`,
			`{"messages":[{"role":"user","content":"q"}],"tool_choice":"required"}`},
		{"toolconfig allowed names", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"toolConfig":{"functionCallingConfig":{"mode":"ANY","allowed_function_names":["lookup"]}}}`,
			`{"messages":[{"role":"user","content":"q"}],"tool_choice":{"type":"function","function":{"name":"lookup"}}}`},
		// generationConfig members fold back to their chat names
		// (vertex_and_google_ai_studio_gemini.py:1090-1133 inverse;
		// responseModalities maps member-wise, VGS:972-985 inverse).
		{"generation config", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"generationConfig":{"temperature":0.5,"top_p":0.9,"top_k":3,"seed":7,"presence_penalty":0.1,"frequency_penalty":0.2,"candidate_count":2,"max_output_tokens":32,"stop_sequences":["END"],"responseLogprobs":true,"logprobs":2,"responseModalities":["TEXT","IMAGE"]}}`,
			`{"messages":[{"role":"user","content":"q"}],"temperature":0.5,"top_p":0.9,"top_k":3,"seed":7,"presence_penalty":0.1,"frequency_penalty":0.2,"n":2,"max_tokens":32,"stop":["END"],"logprobs":true,"top_logprobs":2,"modalities":["text","image"]}`},
		// thinkingConfig buckets back to the effort label the chat wire can
		// say (vertex_and_google_ai_studio_gemini.py:940-970 inverse; the
		// bucket thresholds are the shared effort table).
		{"thinking high", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":4096,"includeThoughts":true}}}`,
			`{"messages":[{"role":"user","content":"q"}],"reasoning_effort":"high"}`},
		{"thinking medium", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":2048,"includeThoughts":true}}}`,
			`{"messages":[{"role":"user","content":"q"}],"reasoning_effort":"medium"}`},
		{"thinking low", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":1024,"includeThoughts":true}}}`,
			`{"messages":[{"role":"user","content":"q"}],"reasoning_effort":"low"}`},
		{"thinking minimal", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":128,"includeThoughts":true}}}`,
			`{"messages":[{"role":"user","content":"q"}],"reasoning_effort":"minimal"}`},
		{"thinking disabled", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":0,"includeThoughts":false}}}`,
			`{"messages":[{"role":"user","content":"q"}],"reasoning_effort":"none"}`},
		// Structured output folds back through the conventional name
		// (vertex_and_google_ai_studio_gemini.py:754-797 inverse); a json
		// mime with no schema can only have come from a bare json_object.
		{"response json schema", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"generationConfig":{"response_mime_type":"application/json","response_json_schema":{"type":"object","properties":{"a":{"type":"string"}}}}}`,
			`{"messages":[{"role":"user","content":"q"}],"response_format":{"type":"json_schema","json_schema":{"name":"response_schema","schema":{"type":"object","properties":{"a":{"type":"string"}}},"strict":false}}}`},
		{"response json only", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"generationConfig":{"response_mime_type":"application/json"}}`,
			`{"messages":[{"role":"user","content":"q"}],"response_format":{"type":"json_object"}}`},
		{"response text plain", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"generationConfig":{"response_mime_type":"text/plain"}}`,
			`{"messages":[{"role":"user","content":"q"}],"response_format":{"type":"text"}}`},
		// serviceTier (transformation.py:1239-1246 inverse).
		{"service tier", `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"serviceTier":"priority"}`,
			`{"messages":[{"role":"user","content":"q"}],"service_tier":"auto"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := ConvertRequest(Gemini, Chat, []byte(tc.gemini))
			if err != nil {
				t.Fatalf("%s: %v", tc.gemini, err)
			}
			if g, w := geminiChatCanonical(t, string(out.Body)), geminiChatCanonical(t, tc.want); g != w {
				t.Fatalf("gemini body %s\n got  %s\n want %s", tc.gemini, g, w)
			}
		})
	}
}

// The documented drops of the reverse request fold: fields the chat wire has
// no spelling for stay behind (spec's inverse table; the loss API names them
// LossNativeParam). Reference round-trip ran during this audit: the forward
// adapters rebuild the body without them.
func TestGeminiChatParityGeminiToChatDropsNativeFields(t *testing.T) {
	body := `{"contents":[{"role":"user","parts":[{"text":"q"}]}],"safetySettings":[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"}],"cachedContent":"caches/1","generationConfig":{"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"V"}}}}}`
	out, err := ConvertRequest(Gemini, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"safetySettings", "cachedContent", "speechConfig"} {
		if strings.Contains(string(out.Body), key) {
			t.Fatalf("native key %q survived the fold: %s", key, out.Body)
		}
	}
	if g, w := geminiChatCanonical(t, string(out.Body)), geminiChatCanonical(t, `{"messages":[{"role":"user","content":"q"}]}`); g != w {
		t.Fatalf("got  %s\n want %s", g, w)
	}
}

// Round-trip: the chat body this codec's reverse fold produces must rebuild
// the original gemini body through a chat->gemini fold that is itself
// reference-exact (the reference round-trip ran during this audit over the
// same corpus; here the composition is exercised in-process). The cases are
// the subset whose reference round-trip is exact - the lossy ones are the
// next test's subject.
func TestGeminiChatParityRequestRoundTrip(t *testing.T) {
	// An unsigned replayed call gains the upstream's skip marker on the way
	// back: the forward fold cannot send a functionCall part with no
	// thoughtSignature (the target refuses it, INVALID_ARGUMENT). The
	// round-trip corpus carries one such call, so the expectation for that
	// body is the marker rather than the original bytes.
	skipMarker := `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"function_call":{"name":"f","args":{"a":1}},"thoughtSignature":"skip_thought_signature_validator"}]},{"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"r"}}}]}]}`
	bodies := []string{
		`{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}`,
		`{"contents":[{"role":"user","parts":[{"text":"q"}]}],"system_instruction":{"parts":[{"text":"be brief"}]}}`,
		`{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"a"}]}]}`,
		skipMarker,
		`{"contents":[{"role":"user","parts":[{"text":"q"}]}],"tools":[{"function_declarations":[{"name":"f1","parameters":{"type":"object"}},{"name":"f2"}]}]}`,
		`{"contents":[{"role":"user","parts":[{"text":"q"}]}],"tools":[{"googleSearch":{}}]}`,
		`{"contents":[{"role":"user","parts":[{"text":"q"}]}],"toolConfig":{"functionCallingConfig":{"mode":"NONE"}}}`,
		`{"contents":[{"role":"user","parts":[{"text":"q"}]}],"toolConfig":{"functionCallingConfig":{"mode":"ANY","allowed_function_names":["lookup"]}}}`,
		`{"contents":[{"role":"user","parts":[{"text":"q"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":4096,"includeThoughts":true},"response_mime_type":"application/json","response_json_schema":{"type":"object","properties":{"a":{"type":"string"}}}}}`,
		`{"contents":[{"role":"user","parts":[{"text":"q"}]}],"serviceTier":"priority"}`,
		// Media-only user contents: the reference forward's blank-text cure
		// appends {"text":" "} (common_utils.py:494-505) - the round trip
		// restores everything but that documented addition.
		`{"contents":[{"role":"user","parts":[{"inline_data":{"data":"AAAB","mime_type":"image/png"}}]}]}`,
	}
	for _, body := range bodies {
		chat, err := ConvertRequest(Gemini, Chat, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", body, err)
		}
		back, err := ConvertRequest(Chat, Gemini, chat.Body)
		if err != nil {
			t.Fatalf("%s -> %s: %v", body, chat.Body, err)
		}
		want := body
		if strings.Contains(body, `"inline_data"`) {
			want = `{"contents":[{"role":"user","parts":[{"inline_data":{"data":"AAAB","mime_type":"image/png"}},{"text":" "}]}]}`
		}
		if g, w := geminiChatCanonical(t, string(back.Body)), geminiChatCanonical(t, want); g != w {
			t.Fatalf("round trip\n orig %s\n chat %s\n got  %s\n want %s", body, chat.Body, g, w)
		}
	}
}

// The lossy round-trips, asserted as documented rather than silently
// different. Both artifacts were reproduced against the reference's own
// forward adapters during this audit:
//
//   - a function_call carrying an id (the wire's stable ids) loses it on the
//     way back: the forward fold does not write ids on the 2.5 line
//     (factory.py:1287-1290, vertex_and_google_ai_studio_gemini.py:278-285);
//   - a signed thought part duplicates on the way back: the reference's own
//     response fold spells a thought part as reasoning_content plus a signed
//     thinking block (vertex_and_google_ai_studio_gemini.py:2307-2322), and
//     the forward fold replays both (transformation.py:880-901), so the
//     composition through chat yields the thought twice.
func TestGeminiChatParityRequestRoundTripDocumentedLosses(t *testing.T) {
	// The id-bearing call: the reverse fold must still read it, so the chat
	// body names it even though the forward fold will not write it back.
	withID := `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"function_call":{"name":"f","args":{"a":1},"id":"native_1"}}]}]}`
	chat, err := ConvertRequest(Gemini, Chat, []byte(withID))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(chat.Body), `"native_1"`) {
		t.Fatalf("the wire's own call id was dropped before the forward fold could lose it: %s", chat.Body)
	}

	// The signed thought part survives the composition with its signature:
	// the forward fold keeps the signature on the thought part (this fold's
	// deviation from the reference), and the residual second thought part
	// comes from the reverse's reasoning_content + thinking_blocks split —
	// the reverse answers a chat READER (reasoning_content is what it
	// displays), so both fields deliberately carry the text; only the
	// thinking_blocks copy rides the next forward fold signed.
	signed := `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"why","thought":true,"thoughtSignature":"SIG"}]}]}`
	chat, err = ConvertRequest(Gemini, Chat, []byte(signed))
	if err != nil {
		t.Fatal(err)
	}
	back, err := ConvertRequest(Chat, Gemini, chat.Body)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"contents":[{"role":"user","parts":[{"text":"q"}]},{"role":"model","parts":[{"text":"why","thought":true},{"text":"why","thought":true,"thoughtSignature":"SIG"}]}]}`
	if g, w := geminiChatCanonical(t, string(back.Body)), geminiChatCanonical(t, want); g != w {
		t.Fatalf("signed thought round trip\n got  %s\n want %s (reasoning for the reader, signed block for the replay)", g, w)
	}
}

// A multi-key arguments object rides the source string's own key order into
// the target's args — executed against the reference's own chat->gemini
// transform during this audit: args {"zeta": 1, "alpha": 2, "mid": {"y":
// true, "x": null}, "list": [3, 1]} kept the stated order. The body marshal
// compacts the bytes (whitespace is not semantic), the ORDER is what a
// decoded map could not give back.
func TestGeminiChatParityMultiKeyArgsOrder(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"q"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"zeta\": 1, \"alpha\": 2, \"mid\": {\"y\": true, \"x\": null}, \"list\": [3, 1]}"}}]}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
	out, err := ConvertRequest(Chat, Gemini, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	const want = `"args":{"zeta":1,"alpha":2,"mid":{"y":true,"x":null},"list":[3,1]}`
	if !strings.Contains(string(out.Body), want) {
		t.Fatalf("multi-key args order not preserved:\n%s", out.Body)
	}
}
