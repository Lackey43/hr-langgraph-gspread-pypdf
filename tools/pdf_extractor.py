from langchain_community.document_loaders import PyPDFLoader



def extract_pdf(file_url : str) -> list:
	"""extracts pdf data"""

	loader = PyPDFLoader(
		file_path=file_url,
		mode="page",
		extraction_mode="plain",

		)

	file = loader.load()

	return [chunk.model_dump(mode="json") for chunk in file]

def display_content(json_data) -> list:


	filtered = [chunk["page_content"] for chunk in json_data]
	return filtered

# if __name__ == "__main__":
# 	file_url = "claude_daigan_resume.pdf"
# 	output = extract_pdf(file_url)
# 	filtered_content = display_content(output)
# 	print(filtered_content)