"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import subprocess
import streamlit as st
from typing import Any, Callable, List, Optional, cast
from langchain_core.tools import BaseTool
from typing_extensions import Annotated
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool, BaseTool

from react_agent.configuration import Configuration


def k8sgpt_analyze() -> str:
    """Run k8sgpt analyze command to analyze the information of the cluster.
    """

    result = subprocess.run(
        ["k8sgpt", "analyze"],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text=True,
    )
    return str(result.stdout)


TOOLS: List[Callable[..., Any]] = [k8sgpt_analyze]
