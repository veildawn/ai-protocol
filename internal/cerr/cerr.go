// Package cerr carries the codec's internal typed errors: the leaf conversion
// packages cannot import the root (it imports them), so the sentinel shapes
// the root maps onto its public error types live here instead of travelling
// as stringly conventions nobody can grep for.
package cerr

import "strconv"

// UnsupportedParam reports a source-dialect parameter the target dialect
// cannot carry. The root maps it onto the public UnsupportedParamError, which
// adds the direction; the leaf knows only the param.
type UnsupportedParam struct {
	Param string
}

func (e UnsupportedParam) Error() string {
	return "unsupported param " + strconv.Quote(e.Param)
}
