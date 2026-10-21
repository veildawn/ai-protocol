// Package chatresponses folds the Chat dialect and the Responses dialect into
// each other: request bodies, response bodies and SSE streams.
//
// The pair is asymmetric on the wire: the Responses dialect names item
// lifecycles explicitly where Chat implies them, so the folds materialize the
// events a Responses client expects and collapse them back on the return leg.
package chatresponses
