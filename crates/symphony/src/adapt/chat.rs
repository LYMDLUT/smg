//! Chat Completions: parser events as stream choices.
//!
//! [`delta`] maps one [`Event`] to at most one [`ChatStreamChoice`]. It is pure: call ids ride on
//! the events, the finish reason is decided from the `Finish` event alone, and nothing is remembered
//! between calls. The driver wraps the choices in the response envelope with the request id, the
//! model, the timestamp, usage and the engine's matched stop, which belong to the response and not
//! to a choice.
//!
//! The shapes are the ones SMG's gateway sends today, so a client cannot tell the two apart:
//!
//! - content and reasoning deltas carry `role: "assistant"`; reasoning goes to `reasoning_content`;
//! - a call begins with its `id`, `type: "function"` and `name` and no arguments; every later
//!   fragment carries the call's `index` and `arguments` only;
//! - the finishing choice has an empty delta without a role, and `finish_reason` is the engine's
//!   reason, except that `stop` after at least one tool call becomes `tool_calls`.
//!
//! Policy this adapter sets, where the events say more than Chat Completions can:
//!
//! - `ReasoningStart`, `ReasoningEnd`, `ToolCallEnd` and `Dropped` have no representation and
//!   produce nothing; the markers a format consumes never reach the client.
//! - `Malformed` text is content: the model wrote it, and the client sees it rather than losing
//!   it, which is what the old driver did on a parser error.
//! - An empty text or an empty argument fragment produces nothing; the gateway sends no empty deltas.

use openai_protocol::{
    chat::{ChatMessageDelta, ChatStreamChoice},
    common::{FunctionCallDelta, ToolCallDelta},
};

use crate::event::{Event, FinishReason};

/// The stream choice one event produces for the choice at `choice`, if it produces one.
pub fn delta(choice: u32, event: &Event) -> Option<ChatStreamChoice> {
    let delta = match event {
        Event::Content(text) | Event::Malformed { text, .. } => {
            if text.text.is_empty() {
                return None;
            }
            ChatMessageDelta {
                content: Some(text.text.clone()),
                ..assistant()
            }
        }
        Event::Reasoning(text) => {
            if text.text.is_empty() {
                return None;
            }
            ChatMessageDelta {
                reasoning_content: Some(text.text.clone()),
                ..assistant()
            }
        }
        Event::ToolCallStart {
            index, id, name, ..
        } => ChatMessageDelta {
            tool_calls: Some(vec![ToolCallDelta {
                index: *index,
                id: Some(id.clone()),
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some(name.clone()),
                    arguments: None,
                }),
            }]),
            ..assistant()
        },
        Event::ToolCallArguments { index, json, .. } => {
            if json.is_empty() {
                return None;
            }
            ChatMessageDelta {
                tool_calls: Some(vec![ToolCallDelta {
                    index: *index,
                    id: None,
                    tool_type: None,
                    function: Some(FunctionCallDelta {
                        name: None,
                        arguments: Some(json.clone()),
                    }),
                }]),
                ..assistant()
            }
        }
        Event::Finish {
            reason, tool_calls, ..
        } => {
            return Some(ChatStreamChoice {
                index: choice,
                delta: ChatMessageDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs: None,
                finish_reason: Some(finish_reason(reason, *tool_calls)),
                matched_stop: None,
            });
        }
        Event::ReasoningStart
        | Event::ReasoningEnd
        | Event::ToolCallEnd { .. }
        | Event::Dropped { .. } => return None,
    };
    Some(ChatStreamChoice {
        index: choice,
        delta,
        logprobs: None,
        finish_reason: None,
        matched_stop: None,
    })
}

/// The stream choices a sequence of events produces, in order.
pub fn deltas<'a>(
    choice: u32,
    events: impl IntoIterator<Item = &'a Event>,
) -> Vec<ChatStreamChoice> {
    events
        .into_iter()
        .filter_map(|event| delta(choice, event))
        .collect()
}

/// The `finish_reason` string for the engine's reason: `stop` after a tool call is `tool_calls`.
fn finish_reason(reason: &FinishReason, tool_calls: u32) -> String {
    match reason {
        FinishReason::Stop if tool_calls > 0 => "tool_calls",
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::Abort => "abort",
        FinishReason::Other(other) => return other.clone(),
    }
    .to_string()
}

/// An assistant delta with nothing in it yet.
fn assistant() -> ChatMessageDelta {
    ChatMessageDelta {
        role: Some("assistant".to_string()),
        content: None,
        tool_calls: None,
        reasoning_content: None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;
    use crate::event::{DropReason, MalformedReason, Text};

    fn wire(choice: Option<ChatStreamChoice>) -> Value {
        serde_json::to_value(choice.expect("a choice")).expect("serializable")
    }

    #[test]
    fn content_is_an_assistant_delta_with_the_text() {
        assert_eq!(
            wire(delta(0, &Event::Content(Text::uncounted("Hello")))),
            json!({
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello", "reasoning_content": null},
                "logprobs": null,
                "finish_reason": null,
            })
        );
    }

    #[test]
    fn reasoning_goes_to_reasoning_content() {
        assert_eq!(
            wire(delta(0, &Event::Reasoning(Text::new("plan", 1)))),
            json!({
                "index": 0,
                "delta": {"role": "assistant", "reasoning_content": "plan"},
                "logprobs": null,
                "finish_reason": null,
            })
        );
    }

    #[test]
    fn a_call_begins_with_its_id_type_and_name_and_no_arguments() {
        let start = Event::ToolCallStart {
            index: 1,
            id: "call_1".into(),
            name: "get_weather".into(),
            source: Text::default(),
        };
        assert_eq!(
            wire(delta(0, &start)),
            json!({
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 1,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather"},
                    }],
                    "reasoning_content": null,
                },
                "logprobs": null,
                "finish_reason": null,
            })
        );
    }

    #[test]
    fn argument_fragments_carry_only_the_index_and_the_arguments() {
        let fragment = Event::ToolCallArguments {
            index: 1,
            json: r#"{"city":"#.into(),
            source: Text::default(),
        };
        assert_eq!(
            wire(delta(0, &fragment)),
            json!({
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{"index": 1, "function": {"arguments": "{\"city\":"}}],
                    "reasoning_content": null,
                },
                "logprobs": null,
                "finish_reason": null,
            })
        );
    }

    #[test]
    fn the_finishing_choice_has_an_empty_delta_and_the_reason() {
        let finish = Event::Finish {
            reason: FinishReason::Length,
            tool_calls: 0,
            reasoning_tokens: 0,
        };
        assert_eq!(
            wire(delta(3, &finish)),
            json!({
                "index": 3,
                "delta": {"reasoning_content": null},
                "logprobs": null,
                "finish_reason": "length",
            })
        );
    }

    #[test]
    fn stop_after_a_tool_call_is_reported_as_tool_calls() {
        let reason = |reason: FinishReason, tool_calls: u32| {
            delta(
                0,
                &Event::Finish {
                    reason,
                    tool_calls,
                    reasoning_tokens: 0,
                },
            )
            .expect("a choice")
            .finish_reason
        };
        assert_eq!(reason(FinishReason::Stop, 0).as_deref(), Some("stop"));
        assert_eq!(reason(FinishReason::Stop, 2).as_deref(), Some("tool_calls"));
        assert_eq!(
            reason(FinishReason::ToolCalls, 0).as_deref(),
            Some("tool_calls")
        );
        assert_eq!(reason(FinishReason::Length, 1).as_deref(), Some("length"));
        assert_eq!(reason(FinishReason::Abort, 0).as_deref(), Some("abort"));
        assert_eq!(
            reason(FinishReason::Other("content_filter".into()), 0).as_deref(),
            Some("content_filter")
        );
    }

    #[test]
    fn malformed_text_reaches_the_client_as_content() {
        let malformed = Event::Malformed {
            text: Text::uncounted("<tool_call>{broken"),
            why: MalformedReason::InvalidArguments,
        };
        let choice = delta(0, &malformed).expect("a choice");
        assert_eq!(choice.delta.content.as_deref(), Some("<tool_call>{broken"));
        assert_eq!(choice.delta.role.as_deref(), Some("assistant"));
    }

    #[test]
    fn events_without_a_chat_representation_produce_nothing() {
        let silent = [
            Event::ReasoningStart,
            Event::ReasoningEnd,
            Event::ToolCallEnd {
                index: 0,
                source: Text::default(),
            },
            Event::Dropped {
                text: Text::uncounted("<think>"),
                why: DropReason::Wrapper,
            },
            Event::Content(Text::uncounted("")),
            Event::Reasoning(Text::uncounted("")),
            Event::ToolCallArguments {
                index: 0,
                json: String::new(),
                source: Text::default(),
            },
        ];
        for event in &silent {
            assert!(delta(0, event).is_none(), "{event:?}");
        }
    }

    #[test]
    fn a_whole_stream_renders_in_order_for_the_given_choice() {
        let events = [
            Event::ReasoningStart,
            Event::Reasoning(Text::uncounted("plan")),
            Event::ReasoningEnd,
            Event::ToolCallStart {
                index: 0,
                id: "call_0".into(),
                name: "get_weather".into(),
                source: Text::default(),
            },
            Event::ToolCallArguments {
                index: 0,
                json: r#"{"city":"Paris"}"#.into(),
                source: Text::default(),
            },
            Event::ToolCallEnd {
                index: 0,
                source: Text::default(),
            },
            Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 1,
                reasoning_tokens: 1,
            },
        ];
        let choices = deltas(2, &events);
        let kinds: Vec<&str> = choices
            .iter()
            .map(|c| {
                match (
                    &c.delta.reasoning_content,
                    &c.delta.tool_calls,
                    &c.finish_reason,
                ) {
                    (Some(_), _, _) => "reasoning",
                    (_, Some(_), _) => "tool_call",
                    (_, _, Some(_)) => "finish",
                    _ => "content",
                }
            })
            .collect();
        assert_eq!(kinds, ["reasoning", "tool_call", "tool_call", "finish"]);
        assert!(choices.iter().all(|c| c.index == 2));
        assert_eq!(choices[3].finish_reason.as_deref(), Some("tool_calls"));
    }
}
