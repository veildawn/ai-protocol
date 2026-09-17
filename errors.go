package protocol

import "fmt"

// UnsupportedParamError is raised when an OpenAI-style parameter cannot be
// represented on the target dialect. Matches LiteLLM default drop_params=false.
type UnsupportedParamError struct {
	Param    string
	From, To Dialect
}

func (e UnsupportedParamError) Error() string {
	return fmt.Sprintf("unsupported param %q converting %s -> %s (set drop_params to omit)", e.Param, e.From, e.To)
}

// ConversionError is a generic failure to read or rebuild a body.
type ConversionError struct {
	Op  string
	Err error
}

func (e ConversionError) Error() string {
	return fmt.Sprintf("protocol %s: %v", e.Op, e.Err)
}

func (e ConversionError) Unwrap() error { return e.Err }
