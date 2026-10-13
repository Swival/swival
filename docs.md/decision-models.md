# Decision Models

A decision model answers questions about a short text with probabilities instead of words.

Swival can give your chat model one as an optional `decide` tool.
The chat model sends the text and its questions, gets numbers back, and keeps working.

Decision models are not chat models.
They always run next to your regular `--model`, never in its place.

## Setup

A hosted decision model needs its name, and its provider's key in the environment:

```sh
export TYPESAFE_API_KEY=...
swival --decision-model typesafe/jev-latest "task"
```

A model on your own server also needs the server's URL:

```sh
swival --decision-model clef-flash-oQ4e \
  --decision-base-url http://127.0.0.1:8000/v1 "task"
```

To use it in every run, put the same settings in a config file:

```toml
decision_model = "clef-flash-oQ4e"
decision_base_url = "http://127.0.0.1:8000/v1"
```

| CLI                   | Config / `Session`    | Purpose                                          |
| --------------------- | --------------------- | ------------------------------------------------ |
| `--decision-model`    | `decision_model`      | Turns on the `decide` tool. Off by default.      |
| `--decision-base-url` | `decision_base_url`   | The server to use.                               |
| `--decision-api-key`  | `decision_api_key`    | Optional key for that server.                    |
| `--no-decisions`      | `decision_model = ""` | Turns off a decision model set in a config file. |

CLI flags win over the project `swival.toml`, which wins over the global config.

These settings are not part of [profiles](customization.md#profiles).
Switching chat models with `/profile` keeps the same decision model.

A name starting with a LiteLLM decision provider, such as `typesafe/`, `cloudflare/`, `openrouter/`, or `hosted_vllm/`, is used as is.
Any other name, even one with a slash like `team/clef`, is treated as a model on your own server and needs `--decision-base-url`.

The URL can be the server root, end with `/v1`, or be the full `/v1/systemone` address.

The decision model never uses the chat model's key.
When you give a URL but no `--decision-api-key`, Swival does not send any key from your environment to that server.
Without a URL, LiteLLM uses the provider's usual environment variable and address.

Swival warns if `decision_api_key` is set in a project config that git tracks.

Unless you pass `--quiet`, Swival prints one line at startup with the decision model and its address.
Nothing is sent to it until the chat model calls the tool.

## The `decide` Tool

`decide` takes the text to judge, called `state`, and some named `questions`:

```json
{
  "state": "The README says install with uv sync.",
  "questions": {
    "mentions_uv": {
      "type": "noul",
      "instructions": "Does the README mention uv sync?"
    },
    "install_tool": {
      "type": "choice",
      "instructions": "Which installation tool is documented?",
      "criteria": {"uv": "uv sync", "pip": "pip install"}
    },
    "clarity": {
      "type": "score",
      "instructions": "How explicit is the instruction?",
      "criteria": ["missing", "ambiguous", "explicit"]
    }
  }
}
```

`state` can be text, a JSON object, or a list.

There are three kinds of questions:

- `noul` gives the probability that the answer is yes.
- `choice` picks one label from `criteria`, which maps each label to a description.
- `score` places the text on the `criteria` scale, listed from lowest to highest. The score can fall between two levels.

The result holds every answer, plus the token count when the server reports one.
Swival sends it as compact JSON; here it is formatted:

```json
{
  "model": "clef-flash-oQ4e",
  "answers": {
    "mentions_uv": {"type": "noul", "noul": 0.93},
    "install_tool": {
      "type": "choice",
      "choice": "uv",
      "confidence": 0.97,
      "probabilities": {"uv": 0.97, "pip": 0.03}
    },
    "clarity": {
      "type": "score",
      "score": 1.77,
      "confidence": 0.86,
      "legend": {"0": "missing", "1": "ambiguous", "2": "explicit"},
      "probabilities": {"0": 0.08, "1": 0.06, "2": 0.86}
    }
  },
  "usage": {"input_tokens": 318, "output_tokens": 0}
}
```

Swival checks every answer against the questions it asked.
A missing answer, an unknown label, a probability outside 0 to 1, probabilities that do not add up to 1,
or a score off the scale makes the call fail.
Swival never fills in a missing answer.

These checks see the answers as LiteLLM returns them, not as the server sent them.
LiteLLM fills in score levels and legend entries the server left out,
so a score answer that skipped some levels can still pass when its probabilities add up to 1.

Like anything else that comes from outside, the result is marked `[UNTRUSTED EXTERNAL CONTENT]` before the chat model sees it.

The chat model uses `decide` on its own when it classifies, sorts, rates, or makes a yes/no call about some text.
You can also ask for it.

The tool only sends what the chat model passes to it.
It never reads files or sends the conversation.

## Size Limit

A request can use up to 15,000 units.
Each byte of the JSON request counts as one unit, each question as 40 more, and each choice label or score level as 10 more.

Questions and labels take more room on the server than their bytes suggest.
When a request is too long, the server quietly cuts off the end of the text, so the extra units keep requests well under its limit.

A request over the limit fails with a message asking the chat model to shorten the text or ask fewer questions.

## Filtering, Encryption, and Costs

Decision requests get the same protections as chat requests.

The [outbound filter](llm-filter.md) sees each request as one user message holding the JSON, with `call_kind` set to `"decision"`.
It can edit or block the request.

[Secret encryption](secrets.md) covers question names, labels, and text.
The answer is decrypted before the chat model sees it.

The size limit is checked after both, just before the request is sent.

Each call shows up like any other tool call: in the terminal, in [reports](reports.md), and in traces.
Reports also list decision requests on their own, under `decision`.

Prices from LiteLLM are only used for a provider's own hosted service.
For your own server, Swival uses the cost the server reports, or marks the call as unpriced.

Calls time out after 30 seconds, or sooner if `--provider-timeout` is lower.
A failed call, a timeout, or a cancellation gives an ordinary `error:` result, so the chat model can explain it and carry on.

For [snapshots](tools.md#snapshot), `decide` counts as a read-only tool.

## Where the Tool Is Available

Subagents, metaskills, the [command provider](providers.md#command-external-program), and the [A2A](a2a.md) and [ACP](acp.md) servers get the decision model too.
Isolated workers, such as the `/audit` and `/simplify` reviewers, never do.

`--network provider-only` allows decision requests, because Swival makes them itself rather than through a sandboxed command.
`--network none` refuses to start with a decision model, because it is a network service.

If you set the chat `--model` to a decision model by mistake, the server refuses to chat, and Swival suggests `--decision-model` instead.
