package protocol

import "strings"

// Dialect is a public wire format. Values match the plan names; Normalize also
// accepts the gateway spellings openai / anthropic. Gemini has no alias: the
// deployment and studio spellings of that wire are not public dialect names,
// so only the dialect's own name normalizes.
type Dialect string

const (
	Chat      Dialect = "chat"
	Messages  Dialect = "messages"
	Responses Dialect = "responses"
	Gemini    Dialect = "gemini"
)

func Normalize(s string) (Dialect, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "chat", "openai":
		return Chat, true
	case "messages", "anthropic":
		return Messages, true
	case "responses":
		return Responses, true
	case "gemini":
		return Gemini, true
	default:
		return "", false
	}
}

func (d Dialect) String() string { return string(d) }
