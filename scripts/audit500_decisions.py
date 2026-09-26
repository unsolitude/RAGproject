"""Explicit source-read assistant decisions; no model predictions used.

Labels concern the complete source, not correctness of a citation number.
None means exclude from this release without rewriting the original claim.
This ledger is not independent human sign-off.
"""

DECISIONS = {}


def add(qid, suffixes, label, reason, evidence=()):
    for suffix in suffixes.split():
        pair_id = f"{qid}_c{int(suffix):02d}"
        if pair_id in DECISIONS:
            raise ValueError(f"Duplicate audit decision: {pair_id}")
        DECISIONS[pair_id] = {
            "label": label, "reason": reason,
            "evidence_ids": [f"passage_{i}" for i in evidence],
        }


# Source 14353: corner folds along the bias, not folding into thirds.
add("12219", "1 6 14", None, "Introductory heading, not a standalone factual proposition.")
add("12219", "2 3 4 5 12 13", None, "Claim refers to named passages or their availability; unsuitable for fixed-gold evidence selection.")
add("12219", "7 8 15", "supported", "Passage 2 specifies a flat surface and the lower-right bias fold.", (2,))
add("12219", "9 10 11", "supported", "Passage 3 gives the remaining three corner folds toward the center along the bias.", (3,))
add("12219", "16", "unsupported", "Corner-fold instructions do not establish folding the quilt into thirds.", (2, 3))

# Source 14355: several sentence fragments end in a broken citation.
add("12233", "1 3 5 7", None, "Heading or detached citation fragment.")
add("12233", "2 4 6", None, "Unclosed citation and gerund list item lacking the object being described; requires a separately identified repaired claim.")
add("12233", "8", None, "Gerund list fragment omits Water Pik as subject; do not silently add query context.")

# Source 14366: citation errors were projected as factual contradictions.
add("12297", "1 4 5 9 10", None, "Heading or detached citation, not an independent claim.")
add("12297", "2 3", "supported", "Passage 1 explicitly gives blue-field removal, collection, peaceful burning and burial.", (1,))
add("12297", "6 8", "supported", "Passage 3 explicitly instructs ceremonial burning and a triangle fold; erroneous answer-side citation does not contradict the factual content.", (3,))
add("12297", "7", None, "'The ceremony that follows' has no content within the isolated claim.")
add("12297", "11", "supported", "Both stated disposal alternatives and the removed blue-field burning occur in the full source.", (1, 3))

# Source 14390: detached symptom labels and unfinished references are not claims.
add("12435", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22", None, "Heading, detached list item or split citation; no complete standalone IBS proposition.")
add("12435", "23", "supported", "The source explicitly says the cause of IBS is unknown and multiple factors are believed involved.", (1, 2))
add("12435", "24 25", None, "Source-scope assertion or inability-to-answer statement.")

add("12471", "1 2 8 14 20", None, "Heading; transfer-direction claim is not present as a standalone instruction here.")
add("12471", "3 7 9 10 11 12 13 21 24 25 26", "supported", "Passage 2 specifies downloading, selecting the audiobook, USB connection and syncing to the iPhone; evaluate instruction content rather than erroneous passage attribution.", (2,))
add("12471", "4 5 22", "supported", "Passage 1 uses desktop OverDrive and requires the iTunes manual music-management setting.", (1,))
add("12471", "6 23", "unsupported", "Source describes transferring from the computer via OverDrive to the Apple device, not a separate transfer from OverDrive to the computer.", (1,))
add("12471", "15 16 17 18 19 27 28", "supported", "Full source gives connection, Devices/Books selection, audiobook sync and Apply/Import controls.", (2, 3))
add("12602", "1", "supported", "Passage 3 identifies Teresa as the Spanish/Portuguese/Italian form and as a female given name.", (3,))
add("12602", "2", "unsupported", "The source presents uncertain alternative etymologies; the claim states their derivation categorically.", (1, 3))
add("12602", "3", None, "Claim is about absence of information in the supplied passages.")
add("12692", "1 2", "supported", "Passage 1 contrasts Vietnam's equitable schooling with India's initial enrollment focus.", (1,))
add("12692", "3", "unsupported", "India's support for Vietnam is described, but reciprocal support of each other's interests is not established.", (2,))
add("12692", "4", None, "Source-scope absence assertion and unresolved 'two countries' reference.")
add("12765", "1", None, "Introductory heading.")
add("12765", "2 3 5", "supported", "Passages 1 and 3 supply the scrap-value, block and residual-depreciation rules as stated.", (1, 3))
add("12765", "4", "unsupported", "Residual depreciation is supported, but the added attribution to Section 50(1) is absent; absence alone is not factual contradiction.", (1, 3))
add("12917", "1 4", "conflict", "The unconditional warming-drawer claim conflicts with the explicit statement that not all ovens have one and some bottom drawers are storage.", (2, 3))
add("12917", "2", None, "'It' lacks an explicit referent in the standalone claim.")
add("12917", "3", None, "Answer heading.")

add("13161", "1", None, "Introductory heading.")
add("13161", "2 3 4", "supported", "Passage 1 specifies room-temperature standing, 375 F, seasoning, rib-side-down placement and one-hour roasting.", (1,))
add("13161", "5 6 8", "supported", "Passage 3 gives pan-on-stove au jus preparation, the 8-10 minute wine step and salt/pepper seasoning.", (3,))
add("13161", "7", "conflict", "The claim assigns the 20-minute half-reduction to wine; the source assigns 8-10 minutes to wine and about 20 minutes to added beef stock.", (3,))
add("13161", "9 10 12 14", "unsupported", "Full source does not specify repeated basting, thermometer checking, 10-15 minute post-roast rest or size-dependent adjustment instructions.", (1, 2, 3))
add("13161", "11", None, "'It' lacks a standalone referent; temperatures would also require unsupported external information.")
add("13161", "13", None, "Source-scope assertion referring to an unspecified set of instructions.")
add("13210", "1 2 3", None, "Broken heading or time-only answer fragment without a standalone chicken-cooking proposition.")
add("13210", "4 5", None, "Explicit passage reference or unresolved 'it mentions' and externally inferred recipe variation.")
add("13366", "1 2", None, "Conversational token or heading.")
add("13366", "3 5", "supported", "Passage 1 provides hot water, mild detergent and dissolved baking soda; hot rinsing also appears in passage 2.", (1, 2))
add("13366", "4 7", None, "Residue/This depends on the preceding claim for its object or action.")
add("13366", "6 8 9", "supported", "Passage 2 provides soft-foam brushing, hot rinsing, lint-free drying and the overnight-red-wine stain warning; judge full-source content rather than citation numbering.", (2,))
add("13366", "10 11 12", "supported", "Passage 3 recommends gentle hand washing, protecting the stem and avoiding chips/spots.", (3,))
add("13442", "1", "supported", "Passages 2 and 3 describe water cleaning, commercial cleaner and brush/broom scrubbing.", (2, 3))
add("13442", "2 3", None, "Source passages give conflicting pressure-washing advice; cannot establish a unique relation without procedural conditions.")
add("13532", "1", "conflict", "The claim assigns ADA a twenty-or-more threshold, whereas passage 1 says four to fifteen; assess source fidelity, not real-world legal correctness.", (1,))
add("13532", "2", "supported", "Passage 1 explicitly contrasts public/private-employer distinctions.", (1,))
add("13532", "3 4", "supported", "Passage 3 contrasts substantial limitation under ADA with limitation under FEHA.", (3,))
add("13917", "1 2 3 4 5 6 7 8 9 10 11", None, "Heading, incomplete symptom label or detached citation; cannot silently supply pregnancy from the query.")
add("13917", "12", None, "'These symptoms' depends on a preceding list outside the claim.")
add("13917", "13", None, "Combines a factual clause with an assertion about absence from supplied passages; exclude source-relative claim.")

add("14284", "1 2 3", None, "All three assertions concern what the supplied passages establish or whether the assistant can answer, rather than a stable factual claim.")
add("14323", "1", "supported", "Passages 1 and 3 describe decorative rollers making random wall textures.", (1, 3))
add("14323", "2", "unsupported", "'Can work better' in passage 1 does not establish that homemade rollers are often more effective.", (1,))
add("14323", "3", None, "Passage-relative assertion blending different roller types; independent object is ambiguous.")
add("14323", "4", "unsupported", "Skin effects are reported for a Dermaroller, not patterned paint rollers; no passage establishes the transferred claim.", (1, 2, 3))
add("14859", "1 7", None, "Introductory heading.")
add("14859", "2 5 6 8", "supported", "Passage 2 supplies the other-island Shugabush, breeding structure and Shugabush/Furcorn combination.", (2,))
add("14859", "3", None, "'This' refers to a preceding acquisition action; cannot independently resolve the asserted procedure.")
add("14859", "4", "conflict", "Source says transfer turns Shugabush into an egg and then hatch it; the claim reverses hatching and egg creation.", (2,))
add("14859", "9", None, "'The monsters'/'desired result' do not name the required breeding pair in this standalone statement.")
add("15008", "1 5 6", "supported", "Passage 3 explicitly defines twins, monozygotic splitting and dizygotic separate eggs/sperm.", (3,))
add("15008", "2", "conflict", "Source says 99% of zygote divisions occur within eight days, not usually between days eight and nine.", (1,))
add("15008", "3", None, "'This' and 'later' require the previous sentence's event and time boundary.")
add("15008", "4", "unsupported", "The listed maternal factors occur in passage 2, but added genetic predisposition is not established by the provided source.", (2,))
add("15010", "1 3", None, "Heading or unresolved 'This' reference.")
add("15010", "2", None, "Internally inconsistent timing (after day nine and within eight days); exclude malformed multi-fact claim rather than count an easy contradiction.")
add("15010", "4 6", "supported", "Passage 3 states the separate-egg and splitting mechanisms.", (3,))
add("15010", "5", "supported", "Passage 2 gives West African descent, age 30-40 and previous pregnancies as factors.", (2,))
add("15011", "1", "supported", "Passage 1 describes post-day-nine splitting as the monoamniotic mechanism; the claim specifies that subtype rather than asserting all twins require it.", (1,))
add("15011", "2", "supported", "Passage 3 explicitly describes dizygotic twins from separate eggs and sperm.", (3,))
add("15011", "3", "supported", "Passage 2 supplies these maternal factors for dizygotic twin likelihood.", (2,))
add("15011", "4 5", None, "Source-scope absence assertion or inability-to-answer statement.")
add("15178", "1 2 3 4", None, "Heading or list predicate with omitted subject (fires); do not use query/previous claim to repair it silently.")
add("15178", "5", "supported", "Passages 2 and 3 link fire to jack-pine seed release, regeneration and competitive advantage.", (2, 3))
add("15201", "1", None, "Introductory heading.")
add("15201", "2", None, "Source itself mixes volcanic and meteorite crater definitions; generic formation comparison has no unambiguous single relation.")
add("15201", "3", "unsupported", "Ranges alone do not establish that calderas are typically larger than craters.", (1, 2, 3))
add("15201", "4", "unsupported", "Source locates calderas at volcano tops but says nothing about lunar or asteroid craters; absence is not contradiction.", (1, 2, 3))
add("15201", "5", "unsupported", "Formation/size information does not establish the asserted generic location difference.", (1, 2, 3))
add("15712", "1", None, "Introductory heading.")
add("15712", "2 4 5", None, "Explicit named-passage assertions excluded consistently with dev50 source-scope policy.")
add("15712", "3", None, "'This' and unspecified price comparison require preceding context.")
add("15712", "6", "unsupported", "The size/tenderloin comparison is between Porterhouse and T-bone, not the separately described sirloin steak.", (1, 2, 3))
add("15716", "1", "supported", "Passage 1 explicitly states Gemini is Latin for twins.", (1,))
add("15716", "2 4", None, "'It' or 'the constellation' omits Gemini in the standalone claim.")
add("15716", "3", "conflict", "Source dates the reference to over 300 years BC, not around 300 years ago.", (1,))
add("17102", "1 3 4 5", "unsupported", "Source mentions a free Google number but not the supplied URL, universal device compatibility, international calls or SIP access via Google; SIP features belong to a separately described cloud service.", (1, 2, 3))
add("17102", "2", None, "'Once there' lacks the referenced service/location in the standalone claim.")
