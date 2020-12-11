#!/usr/bin/env python
import pytest

# https://click.palletsprojects.com/en/7.x/testing/
from click.testing import CliRunner
from smp.cli.main import cli


def test_cli():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 0
    help_result = runner.invoke(cli, ['--help'])
    assert help_result.exit_code == 0
