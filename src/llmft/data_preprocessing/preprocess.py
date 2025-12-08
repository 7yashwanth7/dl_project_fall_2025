# cui_mapping: dict like {"C0041618": "Acquired renal cysts", ...}
# System message for the assistant



def format_data(data, cui_mapping, defaults, include_image_in_messages=False):
    
    # CUIs list for this data
    cuis = data.get("cui", [])
    concepts_text = ", ".join(cuis)

    # CUI Mapping Pairs
    concept_desc_pairs = [
        f"{cui}: {cui_mapping.get(cui, 'UNKNOWN_CUI')}"
        for cui in cuis
    ]
    concept_desc_text = "; ".join(concept_desc_pairs)

    # Assistant text
    assistant_text = (
        f"Caption: {data.get('caption', '')}\n"
        f"Concept descriptions: {concept_desc_text}\n"
        f"Concepts: {concepts_text}"
    )

    user_content = [
        {
            "type": "text",
            "text": defaults["user_prompt"].format(),
        },
        # IMPORTANT:
        # Use placeholder by default to avoid decoding images during .map()
        (
            {"type": "image", "image": data["image"]}
            if include_image_in_messages
            else {"type": "image"}
        ),
    ]

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": defaults["system_message"]}],
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ],
    }
