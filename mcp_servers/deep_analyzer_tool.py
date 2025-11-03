import os
import base64
import sys
from typing import Optional
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.append('..')
import requests
import fitz
from pydantic import Field
from typing import Any
from oxygent import oxy
import logging
logger = logging.getLogger(__name__)

from openai import OpenAI
from mcp_servers.markitdown_tool import MarkitdownConverter



_DEEP_ANALYZER_DESCRIPTION = """A tool that performs systematic, step-by-step analysis or calculation of a given task, optionally leveraging information from external repaths such as attached file or uri to provide comprehensive reasoning and answers.
* At least one of `task` or `path` must be provided. When both are available, the tool will analyze and solve the task in the context of the provided path.
* The `path` can be a local file path or an uri. Support file extensions and uri are as follows:
 - Text: txt, doc, docx, ppt, pptx, csv, pdf, json, jsonl, jsonld, py, pdb, xml...
 - Image: png, jpg, jpeg...
 - Video: mp4, mov...
 - Archive: zip, rar... (NOTE: DO NOT need to unpack the archive, this tool will automatically handle it.)
"""

_DEEP_ANALYZER_INSTRUCTION = """You should step-by-step analyze the task and/or the attached content.
* When the task involves playing a game or performing calculations. Please consider the conditions imposed by the game or calculation rules. You may take extreme conditions into account.
* When the task involves spelling words, you must ensure that the spelling rules are followed and that the resulting word is meaningful.
* When the task involves compute the area in a specific polygon. You should separate the polygon into sub-polygons and ensure that the area of each sub-polygon is computable (e.g, rectangle, circle, triangle, etc.). Step-by-step to compute the area of each sub-polygon and sum them up to get the final area.
* When the task involves calculation and statistics, it is essential to consider all constraints. Failing to account for these constraints can easily lead to statistical errors.

Here is the task:
"""

_DEEP_ANALYZER_SUMMARY_DESCRIPTION = """Please conduct a step-by-step analysis of the outputs from different models. Compare their results, identify discrepancies, extract the accurate components, eliminate the incorrect ones, and synthesize a coherent summary."""


class DeepAnalyzerTool:
    def __init__(self):
        client = OpenAI(
            base_url=os.getenv('DEEPSEEK_URL') ,
            api_key=os.getenv('DEEPSEEK_KEY'),
        )
        client1 = OpenAI(
            base_url=os.getenv('CLAUDE_URL'),
            api_key=os.getenv('CLAUDE_KEY'),
        )
        client4o = OpenAI(
            base_url=os.getenv('OPEN_AI_URL'),
            api_key=os.getenv('OPEN_AI_KEY'),
        )
        
        # self.analyzer_models = {os.getenv('DEEPSEEK_V3'): client, os.getenv('CLAUDE_SONNET'): client1,
        #                         os.getenv('GPT_4O'): client4o}
        self.analyzer_models = {os.getenv('QWEN_VL32B'): client,os.getenv('GLM_4V'): client}

        self.summary_model = client

        self.converter: MarkitdownConverter = MarkitdownConverter(
            use_llm=False,
            timeout=30
        )

    def _analyze(self,
                       client,
                       model_name,
                       task: Optional[str] = None,
                       path: Optional[str] = None) -> str:
        add_note = False
        if not task:
            add_note = True
            task = "Please write a detailed caption for the attached file or uri."

        task = _DEEP_ANALYZER_INSTRUCTION + task
        content = [
            {"type": "text", "text": task},
        ]
        label = True
        if path:

            ext = os.path.splitext(path)[-1].lower()
            logger.info(ext)
            if ext in ['.png', '.jpg', '.jpeg']:
                with open(path, "rb") as image_file:
                    b64_image = base64.b64encode(image_file.read()).decode("utf-8")
                label = False
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    }
                )

                messages = [
                    {
                        "role": 'user',
                        "content": content,
                    }
                ]
                response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False
                        )
            elif ext in ['.pdf']:

                doc = fitz.open(path)
                for page in doc:
                    pix = page.get_pixmap(dpi=300)          # 200 dpi 足够看清
                    png_bytes = pix.tobytes("png")
                    b64_image = base64.b64encode(png_bytes).decode()
                    # 2. 每页当成一张图塞进 content
                    content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    })
                doc.close()

                messages = [
                    {
                        "role": 'user',
                        "content": content,
                    }
                ]
                response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False
                        )
            else:
                extracted_content = self.converter.convert(path).text_content

                content.append(
                    {
                        "type": "text",
                        "text": " - Attached file content: \n\n" + extracted_content,
                    }
                )

                messages = [
                    {
                        "role": 'user',
                        "content": str(content),
                    }
                ]
                response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False
                        )

        else:
            messages = [
                {
                    "role": 'user',
                    "content": str(content),
                }
            ]

            response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False
                        )

        try:
            output = response.choices[0].message.content
            
        except Exception:
            raise Exception(f"Response format unexpected: {response}")

        if add_note:
            output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"
        return output

    def _summarize(self,
                         model,
                         analysis,
                         ) -> str:
        """
        Summarize the analysis and provide a final answer.
        """
        prompt = _DEEP_ANALYZER_SUMMARY_DESCRIPTION

        prompt += "Analysis: \n"
        for model_name, analysis in analysis.items():
            print(analysis)
            prompt += f"{model_name}:\n{analysis}\n\n"

        content = [
            {"type": "text", "text": prompt},
        ]

        messages = [
            {
                "role": 'user',
                "content": str(content),
            }
        ]

        response =  model.chat.completions.create(
            model=os.getenv('DEEPSEEK_V3'),
            messages=messages,
            stream=False,
        )
        try:
            output = response.choices[0].message.content
        except Exception:
            raise Exception(f"Response format unexpected: {response}")

        return output

    def forward(self, task: Optional[str] = None, path: Optional[str] = None):
        """
        Forward the task and/or path to the analyzer model and get the analysis.
        """
        if not task and not path:
            raise ValueError("At least one of task or path should be provided.")

        analysis = {}
        for model_name, client in self.analyzer_models.items():
            analysis[model_name] = self._analyze(client, model_name, task, path)
            logger.info(f"{model_name}:\n{analysis[model_name]}\n")

        summary = self._summarize(self.summary_model, analysis)

        logger.info(f"Summary:\n{summary}\n")

        # Construct the output
        output = "Analysis of models:\n"
        for model_name, analysis in analysis.items():
            output += f"{model_name}:\n{analysis}\n\n"
        output += f"Summary:\n{summary}\n"

        return output


mcp = FastMCP()


@mcp.tool(description="读pdf文件、图片并解析分析")
def deep_analyzer_api(
    task: Any = Field(description="The task to be analyzed and should be solved. If not provided, the tool will focus solely on captioning the attached files or linked URLs."),
    path: Any = Field(description="The attached file or uri to be analyzed. The tool will process and interpret the content of the file or webpage.")
) -> str:
    da_tool = DeepAnalyzerTool()
    print(os.getenv('DEEPSEEK_URL'))
    return da_tool.forward(task, path)


import asyncio

async def main():
    result = await deep_analyzer_api(
        'check order status.',
        r"F:\project\agent\jd\code\test\9e549a4d.jpg"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())