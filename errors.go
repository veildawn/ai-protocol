package protocol

import "fmt"

// UnsupportedParamError is raised when a Chat-style parameter cannot be
// represented on the target dialect and the caller has not opted into omission
// (ConvertOptions.DropParams).
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

// MissingRequiredFieldError is raised when a conversion cannot produce a field
// the target dialect requires and the caller supplied no value for it. The
// codec does not invent values — a required field's worth is host policy — so
// the choice is explicit: pass it through ConvertOptions, or get this error
// instead of a body the upstream would have rejected.
type MissingRequiredFieldError struct {
	Field string
	To    Dialect
}

func (e MissingRequiredFieldError) Error() string {
	return fmt.Sprintf("protocol: target dialect %s requires field %q; supply it via ConvertOptions", e.To, e.Field)
}
