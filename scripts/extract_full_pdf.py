from dotenv import load_dotenv
import asyncio
from pyzerox import zerox

load_dotenv()    
pdf_path = "data/raw/제천시관광정보책자.pdf"

model = "gpt-5.1"
custom_system_prompt = """
Convert the following Korean Travel information PDF page to markdown format.
Allowed only extract the all content text from the PDF page. 
You should not make any changes to the text.
"""
async def main():
    output_dir = "data/processed"  ## 통합된 마크다운 파일을 저장할 디렉토리
    result = await zerox(
        file_path=pdf_path,
        model=model,
        output_dir=output_dir,
        custom_system_prompt=custom_system_prompt,
    )
    return result


# 메인 함수를 실행합니다:
result = asyncio.run(main())