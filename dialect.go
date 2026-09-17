package protocol

import "strings"

// Dialect is a public wire format. Values match the plan names; Normalize also
// accepts the gateway spellings openai / anthropic.
type Dialect string

const (
	Chat      Dialect = "chat"
	Messages  Dialect = "messages"
	Responses Dialect = "responses"
)

func Normalize(s string) (Dialect, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "chat", "openai":
		return Chat, true
	case "messages", "anthropic":
		return Messages, true
	case "responses":
		return Responses, true
	default:
		return "", false
	}
}

func (d Dialect) String() string { return string(d) }
