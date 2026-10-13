package protocol

import (
	"reflect"
	"testing"
)

func TestResponsesToChatConsolidatesSystemMessages(t *testing.T) {
	tests := []struct {
		name     string
		request  jsonMap
		messages []any
	}{
		{
			name: "instructions and input system developer",
			request: jsonMap{
				"instructions": " top rule \n",
				"input": []any{
					map[string]any{"role": "system", "content": "system rule"},
					map[string]any{"role": "user", "content": "first"},
					map[string]any{"type": "message", "role": "developer", "content": []any{
						map[string]any{"type": "input_text", "text": "developer rule"},
						map[string]any{"type": "input_text", "text": "  "},
						map[string]any{"type": "text", "text": "another rule"},
					}},
					map[string]any{"role": "assistant", "content": "reply"},
					map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "result"},
				},
			},
			messages: []any{
				map[string]any{"role": "system", "content": " top rule \n"},
				map[string]any{"role": "system", "content": "system rule"},
				map[string]any{"role": "user", "content": "first"},
				map[string]any{"role": "developer", "content": []any{
					map[string]any{"type": "text", "text": "developer rule"},
					map[string]any{"type": "text", "text": "  "},
					map[string]any{"type": "text", "text": "another rule"},
				}},
				map[string]any{"role": "assistant", "content": "reply"},
				map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "result"},
			},
		},
		{
			name:    "instructions only",
			request: jsonMap{"instructions": "top rule", "input": "hello"},
			messages: []any{
				map[string]any{"role": "system", "content": "top rule"},
				map[string]any{"role": "user", "content": "hello"},
			},
		},
		{
			name: "input system in the middle",
			request: jsonMap{"input": []any{
				map[string]any{"role": "user", "content": "first"},
				map[string]any{"role": "system", "content": "late rule"},
				map[string]any{"role": "assistant", "content": "reply"},
			}},
			messages: []any{
				map[string]any{"role": "user", "content": "first"},
				map[string]any{"role": "system", "content": "late rule"},
				map[string]any{"role": "assistant", "content": "reply"},
			},
		},
		{
			name: "multiple input system developer messages",
			request: jsonMap{"input": []any{
				map[string]any{"role": "developer", "content": "first rule"},
				map[string]any{"role": "user", "content": "hello"},
				map[string]any{"role": "system", "content": "  "},
				map[string]any{"role": "system", "content": "second rule"},
				map[string]any{"role": "developer", "content": "third rule"},
			}},
			messages: []any{
				map[string]any{"role": "developer", "content": "first rule"},
				map[string]any{"role": "user", "content": "hello"},
				map[string]any{"role": "system", "content": "  "},
				map[string]any{"role": "system", "content": "second rule"},
				map[string]any{"role": "developer", "content": "third rule"},
			},
		},
		{
			name: "no instructions",
			request: jsonMap{"instructions": "  ", "input": []any{
				map[string]any{"role": "user", "content": "hello"},
				map[string]any{"role": "system", "content": ""},
			}},
			// Whitespace instructions stay dropped (the deliberate TrimSpace
			// gate), but a system item stating the empty string is content the
			// reference keeps — content-is-not-None, empty string included.
			messages: []any{
				map[string]any{"role": "user", "content": "hello"},
				map[string]any{"role": "system", "content": ""},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := convertReq(t, Responses, Chat, test.request)["messages"]
			if !reflect.DeepEqual(got, test.messages) {
				t.Fatalf("messages=%#v, want %#v", got, test.messages)
			}
		})
	}
}

func TestChatResponsesSystemRoundTrips(t *testing.T) {
	t.Run("chat to responses to chat", func(t *testing.T) {
		original := jsonMap{"messages": []any{
			map[string]any{"role": "user", "content": "first"},
			map[string]any{"role": "system", "content": "first rule"},
			map[string]any{"role": "assistant", "content": "reply"},
			map[string]any{"role": "developer", "content": "second rule"},
			map[string]any{"role": "user", "content": "again"},
		}}
		responses := convertReq(t, Chat, Responses, original)
		// Only a system turn folds into instructions; the developer turn rides
		// the input as its own developer-role item, the way the reference does.
		if responses["instructions"] != "first rule" {
			t.Fatalf("instructions=%v", responses["instructions"])
		}
		if len(responses["input"].([]any)) != 4 {
			t.Fatalf("input=%v", responses["input"])
		}
		chat := convertReq(t, Responses, Chat, responses)
		// Item content that rode as parts stays parts on the chat wire — one
		// text block per part, never joined into a string; the developer turn
		// reads as text (the system/developer fold joins its parts).
		want := []any{
			map[string]any{"role": "system", "content": "first rule"},
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "first"}}},
			map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": "reply"}}},
			map[string]any{"role": "developer", "content": []any{map[string]any{"type": "text", "text": "second rule"}}},
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "again"}}},
		}
		if !reflect.DeepEqual(chat["messages"], want) {
			t.Fatalf("messages=%#v, want %#v", chat["messages"], want)
		}
		again := convertReq(t, Chat, Responses, chat)
		if !reflect.DeepEqual(again, responses) {
			t.Fatalf("second conversion=%#v, want %#v", again, responses)
		}
	})

	t.Run("responses to chat to responses", func(t *testing.T) {
		original := jsonMap{
			"instructions": "top rule",
			"input": []any{
				map[string]any{"role": "user", "content": "hello"},
				map[string]any{"role": "system", "content": "late rule"},
				map[string]any{"role": "assistant", "content": "reply"},
				map[string]any{"role": "developer", "content": "last rule"},
			},
		}
		chat := convertReq(t, Responses, Chat, original)
		responses := convertReq(t, Chat, Responses, chat)
		// Folded back through chat, the system turns become one instruction
		// string; the developer turn keeps its role through both folds and
		// stays its own input item.
		if responses["instructions"] != "top rule late rule" {
			t.Fatalf("instructions=%v", responses["instructions"])
		}
		if len(responses["input"].([]any)) != 3 {
			t.Fatalf("input=%v", responses["input"])
		}
		// The two folds are not symmetric: chat -> responses folds every
		// system turn into one instruction string, while responses -> chat
		// keeps a system or developer item where the caller put it. Folding
		// back therefore yields the merged form, and that form is stable —
		// with item content that rode as parts staying part lists.
		again := convertReq(t, Responses, Chat, responses)
		want := []any{
			map[string]any{"role": "system", "content": "top rule late rule"},
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "hello"}}},
			map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": "reply"}}},
			map[string]any{"role": "developer", "content": []any{map[string]any{"type": "text", "text": "last rule"}}},
		}
		if !reflect.DeepEqual(again["messages"], want) {
			t.Fatalf("second conversion=%#v, want %#v", again["messages"], want)
		}
		twice := convertReq(t, Chat, Responses, again)
		thrice := convertReq(t, Responses, Chat, twice)
		if !reflect.DeepEqual(thrice, again) {
			t.Fatalf("fold is not stable: %#v then %#v", again, thrice)
		}
	})
}
