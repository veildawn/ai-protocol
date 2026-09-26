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
				map[string]any{"role": "system", "content": "top rule\nsystem rule\ndeveloper rule\nanother rule"},
				map[string]any{"role": "user", "content": "first"},
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
				map[string]any{"role": "system", "content": "late rule"},
				map[string]any{"role": "user", "content": "first"},
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
				map[string]any{"role": "system", "content": "first rule\nsecond rule\nthird rule"},
				map[string]any{"role": "user", "content": "hello"},
			},
		},
		{
			name: "no instructions",
			request: jsonMap{"instructions": "  ", "input": []any{
				map[string]any{"role": "user", "content": "hello"},
				map[string]any{"role": "system", "content": ""},
			}},
			messages: []any{map[string]any{"role": "user", "content": "hello"}},
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
		if responses["instructions"] != "first rule\nsecond rule" {
			t.Fatalf("instructions=%v", responses["instructions"])
		}
		if len(responses["input"].([]any)) != 3 {
			t.Fatalf("input=%v", responses["input"])
		}
		chat := convertReq(t, Responses, Chat, responses)
		want := []any{
			map[string]any{"role": "system", "content": "first rule\nsecond rule"},
			map[string]any{"role": "user", "content": "first"},
			map[string]any{"role": "assistant", "content": "reply"},
			map[string]any{"role": "user", "content": "again"},
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
		if responses["instructions"] != "top rule\nlate rule\nlast rule" {
			t.Fatalf("instructions=%v", responses["instructions"])
		}
		if len(responses["input"].([]any)) != 2 {
			t.Fatalf("input=%v", responses["input"])
		}
		again := convertReq(t, Responses, Chat, responses)
		if !reflect.DeepEqual(again, chat) {
			t.Fatalf("second conversion=%#v, want %#v", again, chat)
		}
	})
}
