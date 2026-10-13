// Package dialectvocab carries the tool-type vocabulary the three public
// dialects share. The leaf conversion packages cannot import one another, so
// the names that mean the same thing on every side live here rather than being
// written out once per package.
package dialectvocab

import (
	"encoding/json"
	"strings"
)

// AnthropicHostedTools are the Anthropic-native tool types. Each owns a
// reserved name the provider defines, so a tool of one of these types keeps its
// own shape instead of being rewritten into another dialect's function tool.
var AnthropicHostedTools = []string{
	"web_search", "bash", "text_editor", "code_execution",
	"web_fetch", "memory", "tool_search_tool",
}

// IsAnthropicHostedTool reports whether a tool type names an Anthropic-native
// tool. The type carries a date suffix, so it is matched by prefix.
func IsAnthropicHostedTool(t string) bool {
	for _, hosted := range AnthropicHostedTools {
		if strings.HasPrefix(t, hosted) {
			return true
		}
	}
	return false
}

// ResponsesOnlyTools are the Responses tool types with no equivalent in another
// dialect. Forwarding one makes the target reject the request, so a fold drops
// it rather than passing it through.
var ResponsesOnlyTools = []string{
	"computer_use", "computer_use_preview", "image_generation", "shell", "mcp",
}

// IsResponsesOnlyTool reports whether a tool type exists only on the Responses
// wire.
func IsResponsesOnlyTool(t string) bool {
	for _, only := range ResponsesOnlyTools {
		if t == only {
			return true
		}
	}
	return false
}

// DefaultMaxTokens is the answer budget added to a thinking budget when a
// request asks for thinking but names no output ceiling.
const DefaultMaxTokens = 4096

// ThinkingBudget reads the budget_tokens of an enabled thinking param.
func ThinkingBudget(thinking any) (int, bool) {
	m, ok := thinking.(map[string]any)
	if !ok {
		return 0, false
	}
	if t, _ := m["type"].(string); t != "enabled" {
		return 0, false
	}
	switch n := m["budget_tokens"].(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return 0, false
		}
		return int(i), true
	}
	return 0, false
}
