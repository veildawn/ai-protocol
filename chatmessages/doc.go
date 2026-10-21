// Package chatmessages folds the Chat dialect and the Messages dialect into
// each other: request bodies, response bodies and SSE streams.
//
// The mappings are the reference's own rules for this pair, applied directly.
// Where the reference reaches other dialects by composing through the chat
// dialect, this package speaks both sides natively and never builds an
// intermediate body.
package chatmessages
