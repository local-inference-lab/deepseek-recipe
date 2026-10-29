use deepseek_recipe_core::{
    conversation::Conversation,
    messages::{InputMessage, ToolCall},
    prompt::Prompt,
};
use deepseek_recipe_encoding::{
    PromptEncoding, dsv4::DeepseekV4Encoding, dsv41::DeepseekV41Encoding,
};

const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const USER_SP_TOKEN: &str = "<｜User｜>";
const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
const THINKING_END_TOKEN: &str = "</think>";

fn encoders() -> [Box<dyn PromptEncoding>; 2] {
    [
        Box::new(DeepseekV4Encoding::new()),
        Box::new(DeepseekV41Encoding::new()),
    ]
}

#[test]
fn merged_users_have_one_utf8_byte_boundary_before_the_generation_header() {
    let conversation = Conversation {
        messages: ["café", "你好"]
            .into_iter()
            .map(|content| InputMessage::User {
                content: Prompt::plain(content),
                image_sources: vec![],
            })
            .collect(),
        thinking_mode: false,
        ..Default::default()
    };
    let prefix = format!("{BOS_TOKEN}{USER_SP_TOKEN}café\n\n你好");
    for encoder in encoders() {
        let rendered = encoder.render_conversation(&conversation);
        assert_eq!(
            rendered.prompt,
            format!("{prefix}{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}")
        );
        assert_eq!(rendered.message_boundaries, [BOS_TOKEN.len(), prefix.len()]);
    }
}

#[test]
fn tool_result_boundaries_follow_call_order_and_include_the_closing_wrapper() {
    let calls = ["first", "second"]
        .into_iter()
        .map(|id| ToolCall {
            id: id.into(),
            name: "lookup".into(),
            arguments: "{}".into(),
        })
        .collect();
    let conversation = Conversation {
        messages: vec![
            InputMessage::Assistant {
                content: String::new(),
                reasoning_content: None,
                tool_calls: Some(calls),
                prefix: false,
            },
            InputMessage::Tool {
                content: Prompt::plain("B"),
                image_sources: vec![],
                tool_call_id: Some("second".into()),
            },
            InputMessage::Tool {
                content: Prompt::plain("A"),
                image_sources: vec![],
                tool_call_id: Some("first".into()),
            },
        ],
        thinking_mode: false,
        ..Default::default()
    };
    for encoder in encoders() {
        let rendered = encoder.render_conversation(&conversation);
        let boundaries = &rendered.message_boundaries;
        assert_eq!(boundaries.len(), 4);
        assert_eq!(
            &rendered.prompt[boundaries[1]..boundaries[2]],
            format!("{USER_SP_TOKEN}<tool_result>A</tool_result>")
        );
        assert_eq!(
            &rendered.prompt[boundaries[2]..boundaries[3]],
            "\n\n<tool_result>B</tool_result>"
        );
        assert_eq!(
            &rendered.prompt[boundaries[3]..],
            format!("{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}")
        );
    }
}

#[test]
fn checkpoint_hints_follow_merged_users_and_reordered_tool_results() {
    let marked = |text: &str, at: usize| {
        let mut prompt = Prompt::plain(text);
        prompt.mark_checkpoint(at);
        prompt
    };
    let conversation = Conversation {
        messages: vec![
            InputMessage::User {
                content: marked("café", 3),
                image_sources: vec![],
            },
            InputMessage::User {
                content: marked("second", 3),
                image_sources: vec![],
            },
            InputMessage::Assistant {
                content: String::new(),
                reasoning_content: None,
                tool_calls: Some(vec![
                    ToolCall {
                        id: "a".into(),
                        name: "lookup".into(),
                        arguments: "{}".into(),
                    },
                    ToolCall {
                        id: "b".into(),
                        name: "lookup".into(),
                        arguments: "{}".into(),
                    },
                ]),
                prefix: false,
            },
            InputMessage::Tool {
                content: marked("result B", 8),
                image_sources: vec![],
                tool_call_id: Some("b".into()),
            },
            InputMessage::Tool {
                content: marked("result A", 6),
                image_sources: vec![],
                tool_call_id: Some("a".into()),
            },
        ],
        thinking_mode: false,
        ..Default::default()
    };
    for encoder in encoders() {
        let rendered = encoder.render_conversation(&conversation);
        let prefixes: Vec<_> = rendered
            .checkpoint_hints
            .iter()
            .map(|&i| &rendered.prompt[..i])
            .collect();
        assert_eq!(prefixes.len(), 4);
        assert!(prefixes[0].ends_with("caf"));
        assert!(prefixes[1].ends_with("sec"));
        assert!(prefixes[2].ends_with("result"));
        assert!(prefixes[3].ends_with("result B</tool_result>"));
    }
}
