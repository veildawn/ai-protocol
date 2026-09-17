package stream

// Dialect is a local copy of the root dialect identifiers. The stream package
// cannot import the root package because that would create an import cycle.
type Dialect string

const (
	Chat      Dialect = "chat"
	Messages  Dialect = "messages"
	Responses Dialect = "responses"
)
