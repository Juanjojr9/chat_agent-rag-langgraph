import sys
from datetime import datetime

# Compatibilidad automática zona horaria
if sys.version_info < (3, 9):
    from backports.zoneinfo import ZoneInfo    # pip install backports.zoneinfo tzdata
else:
    from zoneinfo import ZoneInfo

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from typing import Literal

idioma: Literal["es", "en"] = "es"

prompt = (
    ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """Eres un asistente poético y respondes siempre en español.
            Responde con esta estructura:
             Un pareado corto ... y Tu mensaje

            La fecha de hoy es {time}.
            El nombre del usuario es {client}.
            """
        ),
        HumanMessagePromptTemplate.from_template("{messages}"),
    ])
    .partial(
        time=datetime.now(ZoneInfo("Europe/Madrid")).strftime("%A, %d %B %Y, %H:%M"),
        client="JuanJo",
        idioma=idioma,
    )
)

