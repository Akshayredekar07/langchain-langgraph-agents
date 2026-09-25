# from dotenv import load_dotenv
# from langchain.agents import create_agent
# from langchain.agents.middleware import HumanInTheLoopMiddleware
# from langchain_core.runnables import RunnableConfig
# from langchain_nebius import ChatNebius
# from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.types import Command

# load_dotenv()

# model = ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507")


# def send_email(to: str, subject: str, body: str) -> str:
#     """Send an email."""
#     return f"Email sent to {to}: {subject}"


# hitl = HumanInTheLoopMiddleware(
#     interrupt_on={
#         "send_email": {
#             "allowed_decisions": ["approve", "edit", "reject"],
#             "description": "Review this email before sending.",
#         }
#     }
# )

# agent = create_agent(
#     model=model,
#     tools=[send_email],
#     middleware=[hitl],
#     checkpointer=InMemorySaver(),
# )


# def show_messages(result):
#     for message in result.get("messages", []):
#         print(type(message).__name__, ":", message.content)


# def main():
#     config: RunnableConfig = {"configurable": {"thread_id": "hitl-demo"}}

#     result = agent.invoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": (
#                         "Send an email to alice@example.com saying "
#                         "the deployment is complete."
#                     ),
#                 }
#             ]
#         },
#         config=config,
#     )

#     if "__interrupt__" not in result:
#         show_messages(result)
#         return

#     request = result["__interrupt__"][0].value

#     print("HITL request:")
#     for action in request["action_requests"]:
#         print("Tool:", action["name"])
#         print("Args:", action["args"])
#         print("Description:", action.get("description"))

#     decision = input("\nApprove, edit, or reject? ").strip().lower()

#     if decision == "approve":
#         resume = {"decisions": [{"type": "approve"}]}

#     elif decision == "edit":
#         resume = {
#             "decisions": [
#                 {
#                     "type": "edit",
#                     "edited_action": {
#                         "name": "send_email",
#                         "args": {
#                             "to": "alice@example.com",
#                             "subject": "Deployment update",
#                             "body": (
#                                 "The deployment is complete "
#                                 "and ready for verification."
#                             ),
#                         },
#                     },
#                 }
#             ]
#         }

#     elif decision == "reject":
#         resume = {
#             "decisions": [
#                 {
#                     "type": "reject",
#                     "message": "Human rejected the email.",
#                 }
#             ]
#         }

#     else:
#         print("Invalid decision.")
#         return

#     result = agent.invoke(
#         Command(resume=resume),
#         config=config,
#     )

#     print("\nFinal result:")
#     show_messages(result)


# if __name__ == "__main__":
#     main()


from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.runnables import RunnableConfig
from langchain_nebius import ChatNebius
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

model = ChatNebius(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507"
)


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}: {subject}"


hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "send_email": {
            "allowed_decisions": [
                "approve",
                "edit",
                "reject",
            ],
            "description": "Review this email before sending.",
        }
    }
)


agent = create_agent(
    model=model,
    tools=[send_email],
    middleware=[hitl],
    checkpointer=InMemorySaver(),
)


def inspect_interrupt(result):
    print("\n" + "─" * 70)
    print("RESULT INSPECTION")
    print("─" * 70)

    print("\nResult type:")
    print(type(result))

    print("\nResult keys:")
    print(result.keys())

    print("\nFull result:")
    pprint(result)

    interrupts = result.get("__interrupt__", [])

    print("\n" + "─" * 70)
    print("INTERRUPT INSPECTION")
    print("─" * 70)

    print("\nInterrupt collection type:")
    print(type(interrupts))

    print("\nNumber of interrupts:")
    print(len(interrupts))

    for index, interrupt in enumerate(interrupts):
        print("\n" + "-" * 70)
        print(f"INTERRUPT #{index}")
        print("-" * 70)

        print("\nInterrupt type:")
        print(type(interrupt))

        print("\nInterrupt object:")
        print(interrupt)

        print("\nInterrupt ID:")
        print(getattr(interrupt, "id", None))

        print("\nInterrupt value type:")
        print(type(interrupt.value))

        print("\nInterrupt value:")
        pprint(interrupt.value)

        print("\nInterrupt value keys:")

        if isinstance(interrupt.value, dict):
            print(interrupt.value.keys())

        print("\nAction requests:")

        action_requests = (
            interrupt.value.get("action_requests", [])
            if isinstance(interrupt.value, dict)
            else []
        )

        print("Type:", type(action_requests))
        print("Count:", len(action_requests))

        for action_index, action in enumerate(action_requests):
            print("\n" + "." * 60)
            print(f"ACTION REQUEST #{action_index}")
            print("." * 60)

            print("\nAction type:")
            print(type(action))

            print("\nAction:")
            pprint(action)

            if isinstance(action, dict):
                print("\nAction keys:")
                print(action.keys())

                print("\nTool name:")
                print(action.get("name"))

                print("\nTool arguments:")
                pprint(action.get("args"))

                print("\nDescription:")
                print(action.get("description"))

        print("\nAllowed decisions from middleware:")
        print(
            hitl.interrupt_on["send_email"]["allowed_decisions"]
        )


def show_messages(result):
    print("\n" + "─" * 70)
    print("MESSAGES")
    print("─" * 70)

    for index, message in enumerate(
        result.get("messages", [])
    ):
        print(f"\nMESSAGE #{index}")
        print("Type:", type(message))
        print("Content:")
        pprint(message.content)


def main():
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "hitl-debug-demo"
        }
    }

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Send an email to "
                        "alice@example.com saying "
                        "the deployment is complete."
                    ),
                }
            ]
        },
        config=config,
    )

    if "__interrupt__" not in result:
        print("\nNo interrupt occurred.")
        show_messages(result)
        return

    inspect_interrupt(result)

    print("\n" + "─" * 70)
    print("HUMAN DECISION")
    print("─" * 70)

    decision = input(
        "\nApprove, edit, or reject? "
    ).strip().lower()

    if decision == "approve":

        resume = {
            "decisions": [
                {
                    "type": "approve"
                }
            ]
        }

    elif decision == "edit":

        resume = {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "send_email",
                        "args": {
                            "to": "alice@example.com",
                            "subject": "Deployment update",
                            "body": (
                                "The deployment is complete "
                                "and ready for verification."
                            ),
                        },
                    },
                }
            ]
        }

    elif decision == "reject":

        resume = {
            "decisions": [
                {
                    "type": "reject",
                    "message": (
                        "Human rejected the email."
                    ),
                }
            ]
        }

    else:
        print("Invalid decision.")
        return

    print("\n" + "─" * 70)
    print("RESUME COMMAND")
    print("─" * 70)

    print("\nResume payload:")
    pprint(resume)

    print("\nResume payload type:")
    print(type(resume))

    result = agent.invoke(
        Command(resume=resume),
        config=config,
    )

    print("\n" + "─" * 70)
    print("FINAL RESULT")
    print("─" * 70)

    print("\nFinal result type:")
    print(type(result))

    print("\nFinal result keys:")
    print(result.keys())

    pprint(result)

    show_messages(result)


if __name__ == "__main__":
    main()
