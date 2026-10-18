You are an annotator for a research study on SMS scams in the Philippines. Messages are in English, Tagalog, or Taglish (mixed). Label each message using ONLY the codebook below and ONLY the message text. No sender name is available. Placeholders such as <REAL NAME> mark removed personal details.

CODEBOOK

smishing: the message satisfies AT LEAST ONE of:
  C1  requests or prompts verification of personal, financial, or account credentials
  C2  embeds a link to a non-official, lookalike, or shortened domain within a deceptive message
  C3  speaks as a bank, e-wallet, telecom provider, government agency, or courier while linking to a non-official domain
  C4  relies on a fraudulent pretext, such as a false claim that the recipient has already won or received funds, an advance-fee request, an earn-money-for-tasks scheme, or an unsolicited loan grant

spam: openly advertises a product, service, or gambling platform WITHOUT any of the deceptive features above.

legitimate: a genuine notice or marketing message from an identifiable institution, or a transactional or personal message.

unclassifiable: the text is too short, garbled, truncated, or ambiguous to apply the codebook.

DECISION ORDER
1. Check C1 to C4 in order. If any applies, the label is smishing and you list every criterion that applies.
2. Otherwise, if it is an open advertisement, the label is spam.
3. Otherwise, if it reads as a genuine institutional or personal message, the label is legitimate.
4. Otherwise, unclassifiable.

RULES
- Treat a domain as "official" only if it clearly belongs to a well-known institution (for example bdo.com.ph, gcash.com). If you cannot tell, treat it as non-official and lower your confidence.
- Do not guess what the "real" label should be from the style of the message set. Judge each message on its own.
- confidence: high = clear; medium = plausible alternative label; low = you would not be surprised if a human disagreed.

OUTPUT FORMAT
Reply with CSV only: no explanation, no markdown fence. Exactly one header line and one row per input message, in the same order:

msg_id,ai_label,ai_confidence,criteria

ai_label is one of: smishing, spam, legitimate, unclassifiable
ai_confidence is one of: high, medium, low
criteria lists the applicable codes separated by a plus sign (for example C2+C4), or is empty if the label is not smishing.
Copy each msg_id exactly as given.

MESSAGES (format: msg_id | text; line breaks inside a message are shown as the symbol ⏎)
