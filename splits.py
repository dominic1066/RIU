def create_or_append_to_section(sections: map, section_title: str, current_section: str):
    section_title = section_title.strip().lower()
    section = sections.get(section_title)

    if (section):
        # print("adding to existing section:", section_title)
        sections[section_title] += '\n' + current_section
    else:
        # print("creating new section:", section_title)
        sections[section_title] = current_section

def split_by_header_text(text, header_text)->map:
    """
    Splits the given text into sections based on the specified header text.
    
    Parameters:
    text (str): The input text to be split.
    header_text (str): The header text that indicates the start of a new section.
    
    Returns:
    list: A list of sections split by the header text.
    """
    sections = {}
    current_section = ""
    section_title = ""
    
    for line in text.splitlines():
        if line.startswith(header_text):
            # get the text after the header_text
            if (section_title != ""):
                create_or_append_to_section(sections, section_title, current_section)
            section_title = line.partition(header_text)[2]
            current_section = ""
        current_section += '\n' + line
    
    if current_section and section_title != "":
        create_or_append_to_section(sections, section_title, current_section)
    
    return sections

def split_corpus_by_critic(raw_review_corpus)->map:
    result = split_by_header_text(raw_review_corpus, "Reviewed By:")

    return result
