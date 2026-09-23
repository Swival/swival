![Swival Logo](.media/logo.png)

# Swival

A coding agent for any model.

Tell Swival what you want to do, whether that's fixing a bug, cleaning up some code, or building something new.
It reads your files, makes changes, and runs tests, all from your terminal.
You can work with it one step at a time, or give it a task and let it get on with it.

You can use a model running on your own computer or connect to an online service.
Swival is built to work well with smaller models too, so you don't need the biggest model to get useful work done.

As it works, Swival keeps notes on what it's learned and what's left to do, which helps it stay on track during longer jobs.
If you need to stop, you can pick up the session later.
And when you have a bigger task in mind, you can give it a goal and ask it to keep working toward it.

You can also have another model review its changes or ask Swival to check your code for security bugs.
If your work calls for other tools, you can connect those too.

[Learn more at swival.dev](https://swival.dev/).

## Try it

Install with [uv](https://docs.astral.sh/uv/):

```sh
uv tool install --python 3.14 swival
```

Or with Homebrew on macOS:

```sh
brew trust swival/tap
brew install swival/tap/swival
```

Run it from your project directory:

```sh
swival
```

On your first run, Swival offers to help you choose and set up a model.
Then just tell it what you need.

You can also give it a task straight from the command line:

```sh
swival "Find the cause of the failing tests and fix it"
```

If you need help getting started, follow the [setup guide](https://swival.dev/pages/getting-started.html).
You'll find the rest of the documentation at [swival.dev](https://swival.dev/).
