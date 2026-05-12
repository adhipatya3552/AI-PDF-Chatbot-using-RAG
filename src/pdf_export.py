from io import BytesIO

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

def export_chat_to_pdf(messages):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    elements = []

    for message in messages:

        role = (
            "User"
            if message["role"] == "user"
            else "Assistant"
        )

        text = (
            f"<b>{role}:</b> "
            f"{message['content']}"
        )

        elements.append(
            Paragraph(text, styles["BodyText"])
        )

        elements.append(
            Spacer(1, 12)
        )

    doc.build(elements)

    buffer.seek(0)

    return buffer