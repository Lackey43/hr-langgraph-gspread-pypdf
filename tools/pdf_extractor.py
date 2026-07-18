from langchain_community.document_loaders import PyPDFLoader



def extract_pdf(file_url : str) -> list:
	"""extracts pdf data"""

	loader = PyPDFLoader(
		file_path=file_url,
		mode="page",
		extraction_mode="plain",

		)

	file = loader.load()
	# print(file[0])
	output =  [f.page_content for f in file]
	string_output = "\n\n".join(output)
	return string_output
	

