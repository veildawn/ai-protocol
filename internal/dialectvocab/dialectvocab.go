// Package dialectvocab carries the tool-type vocabulary the three public
// dialects share. The leaf conversion packages cannot import one another, so
// the names that mean the same thing on every side live here rather than being
// written out once per package.
package dialectvocab

import "strings"

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
