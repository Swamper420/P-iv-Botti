import unittest

from bot.commands.help_logic import build_help_reply


class HelpLogicTests(unittest.TestCase):
    def test_builds_sorted_unique_command_list(self) -> None:
        reply = build_help_reply(("!sääkuva <kaupunki>", "!help", "aih: <kysymys>", "!help"))
        self.assertEqual(
            reply,
            "Käytettävissä olevat komennot:\n"
            "- `!help`\n"
            "- `!sääkuva <kaupunki>`\n"
            "- `aih: <kysymys>`",
        )


if __name__ == "__main__":
    unittest.main()
