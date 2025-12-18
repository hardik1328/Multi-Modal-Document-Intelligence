import os
import fitz  # PyMuPDF
import io
import base64
from PIL import Image
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.vision_model = "llama-3.2-11b-vision-preview"

    def extract_content(self):
        """
        Extracts text and images from the PDF.
        Returns a list of dictionaries with keys: 'type', 'content', 'page', 'metadata'
        """
        doc = fitz.open(self.pdf_path)
        extracted_data = []

        for page_num, page in enumerate(doc):
            # 1. Extract Text
            text = page.get_text()
            if text.strip():
                extracted_data.append({
                    "type": "text",
                    "content": text,
                    "page": page_num + 1,
                    "metadata": {"source": self.pdf_path, "page": page_num + 1}
                })

            # 2. Extract Images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Verify image is valid
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    if pil_image.width < 50 or pil_image.height < 50: # Skip tiny icons
                        continue
                        
                    # Generate Description
                    description = self._generate_image_description(image_bytes)
                    
                    extracted_data.append({
                        "type": "image",
                        "content": f"[IMAGE DESCRIPTION: {description}]", # Treat as text for embedding
                        "page": page_num + 1,
                        "metadata": {"source": self.pdf_path, "page": page_num + 1, "image_index": img_index}
                    })
                    print(f"Processed image {img_index} on page {page_num + 1}")
                    
                except Exception as e:
                    print(f"Error processing image {img_index} on page {page_num+1}: {e}")

        return extracted_data

    def _generate_image_description(self, image_bytes):
        """
        Uses Groq Vision model to describe the image.
        """
        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail. If it contains a table or chart, summarize the data."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model=self.vision_model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

if __name__ == "__main__":
    # Test run
    processor = PDFProcessor("test/qatar_test_doc.pdf")
    content = processor.extract_content()
    print(f"Extracted {len(content)} items.")
    for item in content[:3]:
        print(item)
