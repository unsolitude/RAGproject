"""Materialize reviewed manual decisions and pending rows for Lesson 3 audit."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/ragtruth/processed/doc_claim_pairs_50.jsonl"
DEST = ROOT / "outputs/lesson3/audit/claims_audit_v7.jsonl"
MANIFEST = ROOT / "outputs/lesson3/audit/claims_audit_v7.manifest.json"
VERSION = "lesson3_claim_audit_v7_reviewed"
AUDITOR = "Codex-assisted review; researcher confirmation pending"

# These IDs were inspected as standalone claims and, where necessary, against
# their source response. All remaining pairs stay pending, not auto-approved.
NONCLAIM = {
    "11957_c01", "11957_c08", "12089_c01", "12639_c01", "12639_c03",
    "12639_c04", "12639_c06", "12639_c07", "12639_c09", "12639_c10",
    "12639_c25", "12675_c01", "12761_c01", "12761_c07", "12837_c01",
    "12837_c07", "12881_c01", "12961_c01", "12961_c10", "13030_c01",
    "13275_c01", "13275_c04", "13275_c05", "13275_c07", "13275_c08",
    "13275_c10", "13275_c11", "13275_c13", "13275_c14", "13275_c16",
    "13275_c17", "13275_c19", "13275_c20", "13275_c21", "14075_c01",
    "14075_c02", "14385_c01", "14385_c02", "14385_c03", "14385_c07",
    "14385_c13", "14385_c18", "14729_c01", "15015_c01", "15059_c01",
    "15268_c01", "15268_c02", "15268_c08", "15491_c01", "15491_c05",
    "15535_c01", "15605_c02", "15605_c03", "15605_c10", "16086_c01",
    "16287_c01", "16287_c09", "17014_c01", "17014_c02", "17014_c04",
    "17014_c14", "17014_c15", "17261_c05", "17409_c01", "17409_c05",
    "17409_c06", "17409_c10", "17409_c11", "17409_c16", "17409_c17",
    "17409_c18", "17409_c19", "17435_c01", "17577_c01", "17621_c01",
    "17621_c02", "17690_c01", "17696_c01", "17577_c08",
}

REVIEWED = {
    "11876_c05": {
        "eligibility": "eligible", "audited_gold_label": "conflict", "decision": "keep",
        "evidence_ids": ["passage_1", "passage_3"],
        "evidence_quote": "Double cream has a higher fat content than single ... Because of the higher fat content it will bind with the flour ...",
        "review_reason": "The source assigns higher fat content and binding to double cream; the claim reverses the comparison to single cream.",
    },
    "11957_c03": {
        "eligibility": "eligible", "audited_gold_label": "supported", "decision": "keep",
        "evidence_ids": ["passage_1", "passage_3"],
        "evidence_quote": "Cut the potatoes into French fries (see photos in post).",
        "review_reason": "The imperative contains a concrete action and object, and the source gives the same instruction.",
    },
    "12675_c07": {
        "eligibility": "mixed", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_1", "passage_2", "passage_3"],
        "evidence_quote": "Gelato, Italy's low fat version of ice cream ... Gelato ... is flavored with fruits, nuts and chocolate.",
        "review_reason": "The sentence joins a broad claim about all three desserts' flavorings with a comparison that attributes stronger flavor to higher fat; the source instead attributes easier tasting to warmer serving temperature. Split and judge separately.",
        "suggested_claims": ["All three desserts can have fruit, nut, and chocolate flavorings.", "Gelato has stronger flavors because of higher fat content."],
    },
    "12639_c11": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_1", "passage_2"],
        "evidence_quote": "as long as you've owned your account for 5 years* and you're age 59½ or older, you can withdraw your money when you want to and you won't owe any federal taxes",
        "review_reason": "The source supports a conditional tax-free withdrawal rule, while the claim combines it with the different assertion that no specific age limit applies. A single unsupported label obscures this distinction.",
        "suggested_claims": ["There is no specific age limit for withdrawing from a Roth IRA.", "After five years and age 59½, Roth IRA withdrawals can be free of federal tax."],
    },
    "17435_c02": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_1", "passage_3"],
        "evidence_quote": "These will be your measuring guides to make the pleated section of the hat.",
        "review_reason": "The claim says there is no information about making a chef hat, but the source contains hat-making steps; it separately claims no two-minute instructions. Split the two absence assertions and adjudicate.",
        "suggested_claims": ["The passages contain no information about making a chef hat.", "The passages contain no instructions for completing a chef hat within two minutes."],
    },
    "16999_c01": {
        "eligibility": "needs_context", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": [], "evidence_quote": None,
        "review_reason": "The isolated organization name lacks a predicate; inspect the adjacent sentence before creating a verifiable claim.",
        "suggested_claims": ["Apple Computer Inc. was established on April 1, 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne."],
    },
    "12775_c04": {
        "eligibility": "needs_context", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": [], "evidence_quote": None,
        "review_reason": "The fragment 'for Arizona, and so on' has no standalone referent; recover the preceding area-code sentence.",
        "suggested_claims": [],
    },
    "12675_c04": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_2"], "evidence_quote": "Gelato is also served at a warmer temperature than ice cream, making the flavors easier to taste.",
        "review_reason": "Gelato's temperature and flavor statement is supported; the claim also asserts a colder sorbet temperature, absent from the source.",
        "suggested_claims": ["Gelato is served warmer than ice cream, making its flavors easier to taste.", "Sorbet is served colder than gelato and ice cream."],
    },
    "12837_c03": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_3"], "evidence_quote": "Mail the completed form to the VA Regional Processing Office in the region of that school's physical address.",
        "review_reason": "Mailing the form to the regional office is supported, but form number 22-0337 and its title are absent from the source.",
        "suggested_claims": ["Complete VA Form 22-0337 for DEA.", "Mail the completed form to the VA Regional Processing Office for the school region."],
    },
    "12961_c06": {
        "eligibility": "eligible", "audited_gold_label": "supported", "decision": "keep",
        "evidence_ids": ["passage_2"], "evidence_quote": "hit Start, type BitLocker, then click Manage BitLocker ... Select the drive ... Turn on BitLocker ... either by password or by smartcard",
        "review_reason": "The source gives the same multi-step BitLocker sequence and unlocking options.",
    },
    "13275_c12": {
        "eligibility": "eligible", "audited_gold_label": "unsupported", "decision": "keep",
        "evidence_ids": [], "evidence_quote": None,
        "review_reason": "All three passages describe video/adapter cables; none supplies the claimed standard audio-cable connection.",
    },
    "14726_c08": {
        "eligibility": "eligible", "audited_gold_label": "supported", "decision": "keep",
        "evidence_ids": ["passage_1"], "evidence_quote": "Remove from oven and let sit for 10 minutes. Then use a metal spatula to gently lift the meatloaf out of the loaf pan to a serving plate.",
        "review_reason": "The source supports the action, timing and utensil; the added baking-dish alternative appears elsewhere in the source.",
    },
    "14729_c08": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_2"], "evidence_quote": "Increase oven temperature to 400 degrees F (200 degrees C), and continue baking 15 minutes",
        "review_reason": "The temperature increase is supported, while its asserted purpose of caramelizing the glaze is absent from the source.",
        "suggested_claims": ["Increase the oven temperature to 400°F (200°C).", "Increasing the temperature caramelizes the glaze."],
    },
    "15268_c06": {
        "eligibility": "eligible", "audited_gold_label": "unsupported", "decision": "keep",
        "evidence_ids": [], "evidence_quote": None,
        "review_reason": "The three passages mention vessel constriction, platelets and coagulation but not chemical signals or platelet-fiber cross-linking.",
    },
    "16477_c01": {
        "eligibility": "eligible", "audited_gold_label": "unsupported", "decision": "keep",
        "evidence_ids": [], "evidence_quote": None,
        "review_reason": "The speeches mention freedoms and taking freedom for granted; they do not compare different groups' experiences or perspectives.",
    },
    "16760_c07": {
        "eligibility": "eligible", "audited_gold_label": "supported", "decision": "keep",
        "evidence_ids": ["passage_1"], "evidence_quote": "Tension rods ... separate lids. Place metal sheets on the inside of cupboards ... Organize necklaces on coat hooks, earrings in ice cube trays, and purses on hangers.",
        "review_reason": "All listed organization examples appear together in the first source passage.",
    },
    "16909_c01": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_3"], "evidence_quote": "Clearwater Beach sea temperatures peak in the range 28 to 31°C ... minimum ... 18 to 21°C",
        "review_reason": "Sea-temperature ranges are supported; the broader claim that Clearwater weather is warm is not established by sea temperatures alone.",
        "suggested_claims": ["Clearwater Beach sea temperatures range from 28–31°C in August and 18–21°C in February.", "The weather in Clearwater is warm."],
    },
    "17014_c11": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_2"], "evidence_quote": "To extend the life of the cat grass it can be transplanted into a larger pot with more soil",
        "review_reason": "Transplanting is supported, but the source does not specify root-boundness as the trigger.",
        "suggested_claims": ["Cat grass can be transplanted into a larger pot with more soil.", "Transplant it when root-bound."],
    },
    "17426_c01": {
        "eligibility": "eligible", "audited_gold_label": "supported", "decision": "keep",
        "evidence_ids": ["passage_1", "passage_2"],
        "evidence_quote": "Place the hens in a little roasting pan and shmear the mixture all over the birds ... Place parsley, rosemary, thyme, garlic, lemon, salt, and pepper in a food processor.",
        "review_reason": "The preparation combines two source passages; ingredients and application locations are preserved.",
    },
    "17621_c08": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_3"], "evidence_quote": "Before applying your paint and primer ...",
        "review_reason": "Painting is in scope, but the asserted roller-or-brush application method is absent from all three source passages.",
        "suggested_claims": ["Apply a coat of paint to the cabinets.", "Use a roller or brush to apply it."],
    },
    "17621_c10": {
        "eligibility": "eligible", "audited_gold_label": "supported", "decision": "relabel",
        "evidence_ids": ["passage_3"], "evidence_quote": "Wait for your initial coat of paint to fully dry before applying your second and, if need be, third coat.",
        "review_reason": "The source explicitly allows additional coats; the automatic unsupported projection does not match this full-source relation.",
    },
    "17621_c11": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_3"], "evidence_quote": "Once your final coat of paint has fully dried, use a clean cloth to wipe a single coat of polyurethane varnish",
        "review_reason": "The varnish step is supported, but the protection and glossy-finish effects are not stated.",
        "suggested_claims": ["Apply polyurethane varnish after the final paint coat dries.", "The varnish protects the surface and makes it glossy."],
    },
    "17621_c12": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_3"], "evidence_quote": "Once the varnish has dried, you may remove your painter's tape.",
        "review_reason": "Removing tape after varnish dries is supported, but the promised crisp, clean edges are not stated.",
        "suggested_claims": ["Remove painter's tape after the varnish dries.", "This reveals crisp, clean edges."],
    },
    "17690_c11": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_3"], "evidence_quote": "exactly the same amount of the same type of hops must be added ... during each boil",
        "review_reason": "Hops are added at the boil; specific bitterness/flavor functions are not fully described by the supplied passages.",
        "suggested_claims": ["Hops are added during the boiling stage.", "Hops provide bitterness and flavor."],
    },
    "17690_c17": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_2"], "evidence_quote": "The brewing process consists of eight key components: ... racking and finishing.",
        "review_reason": "Racking is named in the process list, but its timing, transfer action and purpose are not explained in the source.",
        "suggested_claims": ["Racking and finishing are stages of brewing.", "Racking transfers beer to clarify and stabilize it."],
    },
    "13030_c02": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_1"], "evidence_quote": "what you need to do is find the bass note of the chord",
        "review_reason": "The bass-note instruction appears in the source, but the added piano-foundation explanation does not.",
        "suggested_claims": ["Find the bass note of the chord.", "The bass note is the foundation of the piano chord."],
    },
    "15015_c05": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_1"], "evidence_quote": "a lender may modify the terms of a home loan, allowing the homeowner to sell the property for less than is owed, or transfer the deed back to the lender",
        "review_reason": "Deed transfer is supported, while the explicit acceptance and instead-of-foreclosure conditions are not stated.",
        "suggested_claims": ["A homeowner may transfer the deed back to the lender during loss mitigation.", "The lender accepts the deed instead of pursuing foreclosure."],
    },
    "15015_c06": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_1"], "evidence_quote": "allowing the homeowner to sell the property for less than is owed",
        "review_reason": "The short-sale action is supported, but the stated borrower benefit of avoiding foreclosure is not explicit in the source.",
        "suggested_claims": ["The lender may allow a sale for less than the mortgage balance.", "This sale helps the borrower avoid foreclosure."],
    },
    "15015_c07": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_1"], "evidence_quote": "or transfer the deed back to the lender",
        "review_reason": "Deed transfer is supported; the source does not explicitly describe lender possession or the named deed-transfer procedure.",
        "suggested_claims": ["The homeowner can transfer the deed back to the lender.", "The lender takes possession through a process called deed transfer."],
    },
    "17621_c09": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_3"], "evidence_quote": "Wait for your initial coat of paint to fully dry before applying your second",
        "review_reason": "Waiting for a paint coat to dry is supported, but the instruction to work in sections is absent.",
        "suggested_claims": ["Work in sections when painting cabinets.", "Allow each coat to dry before applying the next."],
    },
    "11957_c09": {
        "eligibility": "eligible", "audited_gold_label": "conflict", "decision": "relabel",
        "evidence_ids": ["passage_2"],
        "evidence_quote": "Bake French fries until light brown and crispy, 40-45 minutes, flipping ¾ of the way through.",
        "review_reason": "The claim says the passages do not mention baking, but passage 2 explicitly describes baking time and end state.",
    },
    "12089_c03": {
        "eligibility": "needs_context", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_1", "passage_2"],
        "evidence_quote": "Transfer the potatoes and pasta to the pan of cabbage ... Put the pizzoccheri, potatoes and Swiss chard in a large saucepan",
        "review_reason": "The first passage transfers cooked pasta into cabbage, while the claim says pasta and cabbage cook together; check the temporal distinction before assigning a relation.",
        "suggested_claims": ["The first passage combines pasta with cabbage after cooking.", "The second passage boils pizzoccheri with potatoes and Swiss chard."],
    },
    "12089_c04": {
        "eligibility": "eligible", "audited_gold_label": "supported", "decision": "keep",
        "evidence_ids": ["passage_3"],
        "evidence_quote": "Stir in the pasta and cook until the pasta is al dente ... Meanwhile ... Add the cabbage",
        "review_reason": "The third passage cooks pasta separately while cabbage is prepared in another pan.",
    },
    "12675_c03": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_2"],
        "evidence_quote": "Gelato, made with skimmed milk and slightly less sugar than ice cream",
        "review_reason": "Gelato-versus-ice-cream sugar is supported; sorbet-versus-gelato sugar is not compared in the sources.",
        "suggested_claims": ["Sorbet has more sugar than gelato.", "Gelato has less sugar than ice cream."],
    },
    "12675_c08": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_1", "passage_2", "passage_3"],
        "evidence_quote": "No dairy is added to sorbet ... Gelato is also served at a warmer temperature ... Sherbet is sorbet's creamier cousin.",
        "review_reason": "The summary bundles dairy, sugar, temperature, texture and flavor; some component comparisons were not supported in the preceding list.",
    },
    "12837_c02": {
        "eligibility": "needs_context", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_2"],
        "evidence_quote": "The following guide will help you determine if you are eligible ... eligible dependents of certain veterans",
        "review_reason": "The source describes a guide and a broad eligibility group but the clipped passage does not list actionable eligibility requirements.",
    },
    "12837_c04": {
        "eligibility": "eligible", "audited_gold_label": "unsupported", "decision": "relabel",
        "evidence_ids": [], "evidence_quote": None,
        "review_reason": "Passage 3 says to check below for a post-office-box address, but the supplied source text ends before any address appears.",
    },
}

for _pid in ("12639_c12", "12639_c14", "12639_c16", "12639_c18", "12639_c20", "12639_c22", "12639_c24"):
    REVIEWED[_pid] = {
        "eligibility": "needs_context", "audited_gold_label": "needs_adjudication",
        "decision": "review_and_split", "evidence_ids": [], "evidence_quote": None,
        "review_reason": "The isolated '(Not mentioned in the passages)' parenthetical depends on the preceding numbered claim and is not independently interpretable.",
        "suggested_claims": [],
    }

REVIEWED.update({
    "12089_c02": dict(eligibility="eligible", audited_gold_label="supported", decision="keep", evidence_ids=[], evidence_quote=None, review_reason="All three passages describe cooking with prepared pizzoccheri, not making the pasta dough itself; this is a source-bounded absence claim."),
    "12089_c05": dict(eligibility="eligible", audited_gold_label="supported", decision="keep", evidence_ids=[], evidence_quote=None, review_reason="The supplied recipes use existing pasta and provide no from-scratch pizzoccheri pasta recipe."),
    "13030_c05": {
        "eligibility": "eligible", "audited_gold_label": "conflict", "decision": "relabel",
        "evidence_ids": ["passage_1"], "evidence_quote": "a 9th chord has 5 notes, but the guitar can only play four different notes at a time. The piano has all five",
        "review_reason": "The claim reverses which instrument has the note-limit constraint.",
    },
    "13030_c06": {
        "eligibility": "eligible", "audited_gold_label": "conflict", "decision": "relabel",
        "evidence_ids": ["passage_1"], "evidence_quote": "notes need to be subtracted from chords on a guitar because of string organization ... The piano has all five",
        "review_reason": "The source describes note subtraction for guitar limitations, not for a piano-friendly chord.",
    },
    "13950_c02": {
        "eligibility": "mixed", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_1", "passage_2", "passage_3"],
        "evidence_quote": "Additional ounces will cost $0.22 ... The price of each additional ounce is 21 cents",
        "review_reason": "The sources disagree across rate periods; the undated claim needs a date before relation judgment.",
        "suggested_claims": ["In the announced 2015 schedule, each additional ounce costs $0.22."],
    },
    "13950_c03": {
        "eligibility": "mixed", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_1", "passage_3"],
        "evidence_quote": "Postcard rates will increase ... to $0.35, from $0.34 in 2014 ... It will cost you 34 cents to send a domestic postcard",
        "review_reason": "The source passages mix two historical rate periods; the undated claim needs a date.",
        "suggested_claims": ["Under the announced 2015 rates, domestic postcard postage is $0.35."],
    },
    "14385_c17": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_2", "passage_3"],
        "evidence_quote": "Then multiply them to get the main area of the room ... Subtract the surface area of doors ... when calculating how much paint you'll need",
        "review_reason": "Main floor area is supported, but deriving wall-paint quantity from it conflates floor and wall areas.",
    },
    "14729_c07": {
        "eligibility": "needs_context", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_2", "passage_3"],
        "evidence_quote": "combine the brown sugar, mustard and ketchup ... Bake ... for 1 hour ... remaining tomato sauce and ketchup ... continue baking 10 minutes",
        "review_reason": "The claim combines an initial mustard glaze with a different last-ten-minute tomato sauce step.",
    },
    "15015_c04": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_2"],
        "evidence_quote": "The most common modifications are lowering the interest rate and extending the term to up to 40 years",
        "review_reason": "Loan modification types are supported; helping catch up missed payments is not explicit.",
    },
    "15015_c08": {
        "eligibility": "eligible", "audited_gold_label": "unsupported", "decision": "relabel",
        "evidence_ids": [], "evidence_quote": None,
        "review_reason": "The sources do not state the asserted borrower-by-borrower and lender-policy variation.",
    },
    "15059_c05": {
        "eligibility": "needs_split", "audited_gold_label": "needs_adjudication", "decision": "review_and_split",
        "evidence_ids": ["passage_2", "passage_3"],
        "evidence_quote": "design, build ... maintain infrastructure ... Consider construction costs, government regulations, potential environmental hazards",
        "review_reason": "Core civil-engineering duties are supported; the safe, efficient and compliant outcome clause needs separate verification.",
    },
})

# These groups were read claim by claim alongside their complete three-passage
# sources. A shared excerpt identifies the source region for related claims.
REVIEWED.update({
    "15268_c13": dict(eligibility="eligible", audited_gold_label="unsupported", decision="relabel", evidence_ids=[], evidence_quote=None, review_reason="The passages describe a loose platelet plug and coagulation cascade but do not mention cross-linking of platelet fibers."),
    "15491_c02": dict(eligibility="eligible", audited_gold_label="conflict", decision="relabel", evidence_ids=["passage_3"], evidence_quote="Learning how to give a proper hickey", review_reason="The claim denies that the passages explain how to leave a hickey; passage 3 explicitly introduces such instructions."),
    "15491_c04": dict(eligibility="needs_split", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=["passage_3"], evidence_quote="Learning how to give a proper hickey", review_reason="The first clause acknowledges how-to content, whereas the second denies instructions on leaving a hickey. The self-contradictory pair should be split.", suggested_claims=["Passage 3 describes how to give a proper hickey.", "Passage 3 does not explain how to leave one."]),
    "15605_c01": dict(eligibility="needs_context", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=[], evidence_quote=None, review_reason="The passages give some retirement-plan facts, while 'comprehensive information' has no defined coverage threshold; this scope judgment needs researcher adjudication."),
    "15864_c05": dict(eligibility="needs_context", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=["passage_1", "passage_2"], evidence_quote="SIM swap ... less than an hour ... port ... usually within 48 hours", review_reason="The hour-scale timing belongs to SIM swapping, while the answer is about number porting; resolve the referent of 'the process' before labeling."),
    "16363_c03": dict(eligibility="needs_split", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=[], evidence_quote=None, review_reason="The first clause asserts absence of tax-return-copy information; the second is a refusal rather than a factual answer. Audit them separately.", suggested_claims=["The passages do not mention a copy of a tax return."]),
    "16477_c05": dict(eligibility="mixed", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=["passage_1", "passage_2"], evidence_quote="Sometimes we fail to hear or heed these voices of freedom ... four essential human freedoms", review_reason="The source does not establish varied group experiences, but it does omit specific group perspectives; separate the unsupported inference from the source-absence observation.", suggested_claims=["Different groups have varied experiences of the four freedoms.", "The passages do not give specific perspectives for these groups."]),
    "17014_c07": dict(eligibility="needs_split", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=["passage_2"], evidence_quote="run the water through the container once a day ... DO NOT let the grass sit in standing water", review_reason="Watering and avoiding standing water are sourced, but constant soil moisture and thorough watering are not explicit.", suggested_claims=["Water cat grass daily or more often if necessary.", "Keep soil consistently moist but not waterlogged."]),
    "17014_c12": dict(eligibility="needs_split", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=["passage_2"], evidence_quote="Fresh is Best ... the grass needs water and natural light to grow", review_reason="Natural light and water are sourced; 'fresh water' and optimal cat nutrition conflate freshness of grass with freshness of water.", suggested_claims=["Cat grass needs water and natural light to grow.", "Fresh water provides optimal nutrition for the cat."]),
    "17435_c03": dict(eligibility="mixed", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=["passage_1", "passage_2", "passage_3"], evidence_quote="These will be your measuring guides to make the pleated section of the hat", review_reason="The passages do include cake and hat-making material; whether they suffice for the two-minute chef-hat question is subjective and must be separated.", suggested_claims=["The passages contain cake and pleated-hat instructions.", "They do not contain enough information to make a chef hat in two minutes."]),
    "17621_c06": dict(eligibility="needs_split", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=["passage_3"], evidence_quote="Before applying your paint and primer", review_reason="Primer use is mentioned, but the source does not instruct full-surface coverage.", suggested_claims=["Apply primer to the cabinets.", "Cover the entire surface with primer."]),
    "17690_c15": dict(eligibility="mixed", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=["passage_1", "passage_2"], evidence_quote="If brewing an ale ... pitch ale yeast ... cooling, fermentation", review_reason="Ale yeast pitching and the cooling/fermentation stages are mentioned; lager yeast and transfer details are not supplied.", suggested_claims=["For ale, pitch ale yeast.", "After cooling, inoculate wort with either ale or lager yeast."]),
    "17696_c07": dict(eligibility="eligible", audited_gold_label="unsupported", decision="relabel", evidence_ids=[], evidence_quote=None, review_reason="The source describes placing vegetables on foil squares but not covering packets with a second foil sheet."),
    "17261_c01": dict(eligibility="needs_split", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=["passage_1", "passage_2", "passage_3"], evidence_quote="Pot roast isn't really a specific recipe or cut of meat ... Beef Manhattan ... roast beef", review_reason="The response combines a source-bounded absence of a named cut with a refusal to answer; only the former is a verifiable proposition.", suggested_claims=["The passages do not specify a cut of beef for Beef Manhattan."]),
})

for _pid, _name in {
    "16287_c10": "Bread of life (John 6:48)",
    "16287_c11": "Light of the world (John 8:12)",
    "16287_c13": "Door (John 10:9)",
}.items():
    REVIEWED[_pid] = dict(eligibility="needs_context", audited_gold_label="needs_adjudication", decision="review_and_split", evidence_ids=[], evidence_quote=None, review_reason=f"The list fragment '{_name}' lacks a standalone predicate; restore the preceding heading and check the verse before evaluation.")

MANUAL_KEEP_GROUPS = [
    (("11876_c01", "11876_c02", "11876_c03", "11876_c04", "11876_c06", "11876_c07"),
     ["passage_1", "passage_2", "passage_3"],
     "Single cream ... 10 percent and 12 percent butterfat ... double cream ... 48 percent ... single cream is too thin to be whipped ... Double cream has a higher fat content",
     "The source supplies the fat-content comparison, texture and whipping distinctions used by these claims."),
    (("11957_c02", "11957_c04", "11957_c05", "11957_c06", "11957_c07"),
     ["passage_1", "passage_2", "passage_3"],
     "Pre-heat the oven to 400 degrees Fahrenheit ... Line a large cookie sheet with parchment paper ... Bake French fries until light brown and crispy, 40-45 minutes",
     "Each instruction matches the source's explicit baked-fries recipe."),
    (("12639_c02", "12639_c05", "12639_c08"),
     ["passage_2", "passage_3"],
     "Withdrawals of Roth IRA contributions are always both tax-free and penalty-free ... first-time home buyer ... up to $10,000 in earnings",
     "The source states the contribution, early-earnings and home-buyer conditions."),
    (("12675_c02", "12675_c06"),
     ["passage_1", "passage_2", "passage_3"],
     "No dairy is added to sorbet ... Gelato, made with skimmed milk and slightly less sugar than ice cream ... Sherbet is sorbet's creamier cousin",
     "The dairy and sherbet-texture relations are explicitly present in the source."),
    (("12761_c02", "12761_c03", "12761_c04", "12761_c05", "12761_c06", "12761_c08", "12761_c09", "12761_c10", "12761_c11"),
     ["passage_1", "passage_2", "passage_3"],
     "The Vitamin E found in coconut oil soothes eczema, sunburn and psoriasis ... its antiviral and antifungal benefits ... Enhances cognitive thinking ... moisturizing benefits",
     "The benefit statements are in the supplied passages; the Livestrong qualifier is absent from them as the last claim says."),
    (("12775_c01", "12775_c02", "12775_c03"),
     ["passage_1", "passage_3"],
     "area codes are not assigned based upon population ... non-geographic area codes ... Alabama Area Codes: 205 - 251 - 256 - 334",
     "The source states the assignment caveat and listed non-geographic and state code examples."),
    (("12881_c02", "12881_c03", "12881_c04", "12881_c05", "12881_c06", "12881_c07", "12881_c08", "12881_c09", "12881_c10", "12881_c11", "12881_c12"),
     ["passage_2", "passage_3"],
     "Fill the bottom of the pan with a layer of water ... Add the vinegar ... Remove the pan from the heat and add the baking soda ... scour as normal ... make a paste ... rinse under water",
     "The vinegar-and-baking-soda cleaning sequence and follow-up steps appear across passages 2 and 3."),
    (("12961_c02", "12961_c03", "12961_c04", "12961_c05", "12961_c07", "12961_c08", "12961_c09"),
     ["passage_1", "passage_2", "passage_3"],
     "Choose Encryption ... click Encrypt tab ... folder Properties ... hit Start, type BitLocker ... save the recovery key ... encrypt the entire drive, or only the used space",
     "The offline-file, folder-permission and BitLocker actions are described in the source."),
    (("13030_c07", "13030_c08"),
     ["passage_3"],
     "here’s how to translate guitar tabs so you can play piano chords ... The notes of the open strings ... each fret on guitar is a half step",
     "The source explains a basic tablature-to-piano procedure."),
    (("13275_c02", "13275_c03", "13275_c06", "13275_c09", "13275_c15", "13275_c18"),
     ["passage_1", "passage_2", "passage_3"],
     "adapter cable to connect your computer's USB port to your TV's HDMI port ... different ... need an adapter cable ... S-Video or component video cable ... Turn on the TV and use the remote control to select Input",
     "The video-cable, adapter and input-selection steps appear in the source."),
    (("13950_c01", "13950_c04"),
     ["passage_1", "passage_2"],
     "a one ounce First Class Letter ... remain $0.49 ... an international destination ... to $1.20",
     "The quoted rates agree with the announced 2015 rate schedule in these passages."),
    (("14075_c03", "14075_c04", "14075_c05", "14075_c06", "14075_c07", "14075_c08", "14075_c09"),
     ["passage_1", "passage_2", "passage_3"],
     "soil warms to 50 degrees F ... germinate within 10 days ... water your carrots twice a day ... plant the carrot seeds with radish seeds",
     "The temperature, timing, watering and radish instructions are given in the source."),
    (("14244_c01", "14244_c02", "14244_c03", "14244_c04", "14244_c05"),
     ["passage_3"],
     "valve/hinge mechanism ... reacts via a float mechanism and closes ... prevents water from filling the tube",
     "The dry-snorkel mechanism and water prevention are described directly."),
    (("14385_c04", "14385_c05", "14385_c06", "14385_c08", "14385_c09", "14385_c10", "14385_c11", "14385_c12", "14385_c14", "14385_c15", "14385_c16"),
     ["passage_1", "passage_2", "passage_3"],
     "Multiply the length and the width ... 12 feet wide and 12 feet long ... 144 square feet ... Subtract the surface area of doors ... Record the length and width",
     "The room-measurement formulas, examples and recording steps are in the source."),
    (("14726_c01", "14726_c02", "14726_c03", "14726_c04", "14726_c05", "14726_c06", "14726_c07", "14726_c09", "14726_c10"),
     ["passage_1", "passage_2", "passage_3"],
     "Preheat oven to 350 degrees F ... combine the beef, egg, onion, milk and bread ... Bake for 1 hour ... 155°F ... Increase oven temperature to 400 degrees F",
     "The preparation, bake and optional finishing steps follow the source recipes."),
    (("14729_c02", "14729_c03", "14729_c04", "14729_c05", "14729_c06"),
     ["passage_1", "passage_3"],
     "Preheat oven to 350 degrees F ... Bake for 1 hour ... Remove from oven and let sit for 10 minutes ... metal spatula",
     "The source supports the bake and removal sequence."),
    (("14790_c01", "14790_c02", "14790_c03", "14790_c04"),
     ["passage_1", "passage_2", "passage_3"],
     "A registered trademark is designated with the symbol ® ... guilty of trademark infringement ... unregistered trademark ... common law rights",
     "The registration, infringement and unregistered-mark distinctions are in the source."),
    (("14959_c01", "14959_c02", "14959_c03", "14959_c04", "14959_c05"),
     ["passage_1", "passage_2", "passage_3"],
     "Sprinkle bagels with water and wrap in foil ... 350 degrees for 10 to 15 minutes ... refrigerating ... stale ... Frozen, they will keep for 3-4 months",
     "The bagel refresh, refrigeration and freezing advice is in the source."),
    (("15015_c03", "15015_c09"),
     ["passage_1", "passage_2", "passage_3"],
     "Loss mitigation is a process used by mortgage lenders ... new rules ... provide consistent and meaningful protections for borrowers ... industry necessary flexibility",
     "The source describes mortgage loss mitigation and the CFPB borrower-protection context."),
    (("15059_c02", "15059_c03", "15059_c04"),
     ["passage_2", "passage_3"],
     "civil engineers conceive, design, build ... maintain infrastructure ... Analyze long range plans ... Consider construction costs, government regulations, potential environmental hazards",
     "The listed civil-engineering duties correspond to the source descriptions."),
    (("15268_c03", "15268_c04", "15268_c05", "15268_c07", "15268_c09", "15268_c10", "15268_c11", "15268_c12", "15268_c14"),
     ["passage_1", "passage_2", "passage_3"], "injured blood vessel constricts ... platelets ... loose platelet plug ... coagulation factors", "The vessel, platelet, cascade and clotting steps are described in the hemostasis sources."),
    (("15336_c01", "15336_c02", "15336_c03", "15336_c04", "15336_c05", "15336_c06"),
     ["passage_1", "passage_2", "passage_3"], "phosphorus ... bones and teeth ... calcium ... kidneys ... energy ... muscle pain", "The mineral's amount, roles and balance effects correspond to the three source passages."),
    (("15415_c01", "15415_c02", "15415_c03"),
     ["passage_1", "passage_2", "passage_3"], "Convert a USB connected printer to Wireless ... Printers and Faxes ... Add Printer Wizard", "The Windows XP network, installation and Add Printer steps are present in the source."),
    (("15491_c03",), ["passage_1", "passage_2"], "home remedies ... what not to do when giving a hickey", "The first two passages cover remedies and precautions as the claim says."),
    (("15535_c02", "15535_c03", "15535_c04", "15535_c05", "15535_c06", "15535_c07"),
     ["passage_1", "passage_2", "passage_3"], "electrical impulse ... hormones travel through the blood ... long-lasting effects", "The neural-versus-hormonal transmission, duration and target distinctions match the source."),
    (("15852_c01", "15852_c02", "15852_c03", "15852_c04", "15852_c05"),
     ["passage_1", "passage_2", "passage_3"], "olive ... between two tapered surfaces ... fitting and the nut ... all components ... new", "The compression-fitting mechanics and replacement warning are in the source."),
    (("15864_c01", "15864_c02", "15864_c03", "15864_c04", "15864_c06"),
     ["passage_1", "passage_2", "passage_3"], "join O2 ... PAC ... original mobile number and the temporary mobile number ... within 48 hours", "The O2 account, PAC, number and transfer instructions are source-backed; SIM-swap timing is separately held for adjudication."),
    (("16099_c01",), ["passage_1", "passage_2", "passage_3"], "fluid in the knee ... swelling ... difficult to bend", "The source attributes knee swelling and restricted bending to fluid accumulation."),
    (("16287_c02", "16287_c03", "16287_c05"), ["passage_1", "passage_2", "passage_3"], "I am the bread of life ... I am the light of the world ... I am the door", "These three verse quotations occur in the supplied passages."),
    (("16363_c01", "16363_c02"), ["passage_1", "passage_2"], "P60 ... end of the tax year ... pay and deductions", "The source describes the P60's timing and contents."),
    (("16434_c01", "16434_c02", "16434_c03"), ["passage_1", "passage_2", "passage_3"], "CO2 sink ... 40% of the oceanic intake ... slowing down ... climate change", "The Southern Ocean sink and possible weakening match the source passages."),
    (("16477_c02", "16477_c03", "16477_c04"), ["passage_1", "passage_2", "passage_3"], "privilege of our freedom ... freedom of speech and expression ... freedom ... worship", "The source states the freedom themes; passages 2 and 3 repeat the same text and offer no group-specific account."),
    (("16760_c01", "16760_c02", "16760_c03", "16760_c04", "16760_c05", "16760_c06"), ["passage_1", "passage_2", "passage_3"], "Sort items by function ... donate pile ... sort items by room and location", "The source directly gives the function, donation and room-organization steps."),
    (("16765_c01", "16765_c02", "16765_c03", "16765_c04"), ["passage_1", "passage_2", "passage_3"], "cheesecloth ... Cut the stem 12 inches ... suspend the sunflower ... fine-mesh screen", "The sunflower-seed drying sequence follows the source steps."),
    (("16909_c02",), ["passage_3"], "rash vest and board shorts ... minimum ... fully sealed wetsuit", "The surfing clothing and February wetsuit caveat are stated in passage 3."),
    (("16999_c02", "16999_c03", "16999_c04", "16999_c05"), ["passage_1", "passage_2", "passage_3"], "established on April 1, 1976 ... Apple I ... incorporated January 3, 1977 ... launched the Macintosh", "The establishment, products, incorporation and Macintosh chronology are sourced."),
    (("17014_c03", "17014_c05", "17014_c08", "17014_c09", "17014_c10", "17014_c13"), ["passage_2", "passage_3"], "drainage holes ... moist potting soil ... water ... natural light ... refrigerated ... standing water", "The cat-grass container, light, watering, cooling and mold precautions are in the source."),
    (("17261_c02", "17261_c03", "17261_c04"), ["passage_1", "passage_2", "passage_3"], "Pot roast isn't really a specific ... Beef Manhattan ... roast beef ... Put Roast in Crock pot", "Each passage discusses roast without identifying a specific beef cut."),
    (("17347_c01", "17347_c02", "17347_c03", "17347_c04", "17347_c05", "17347_c06", "17347_c07", "17347_c08", "17347_c09", "17347_c10", "17347_c11", "17347_c12", "17347_c13", "17347_c14"), ["passage_1", "passage_2", "passage_3"], "Boil sampalok ... saut garlic and onion ... add pork ... eggplant ... water spinach ... Serve with rice", "The fourteen cooking instructions match the sinigang source recipes and their stated timings."),
    (("17409_c02", "17409_c03", "17409_c04", "17409_c07", "17409_c08", "17409_c09", "17409_c12", "17409_c13", "17409_c14", "17409_c15"), ["passage_1", "passage_2", "passage_3"], "Fold both tails ... Cross the ribbon ... mountains and valleys ... plastic zip tie ... Hot glue", "The bow folding, creasing, tie and gluing steps are directly stated."),
    (("17426_c02", "17426_c03"), ["passage_1"], "Roast, breast side down for 25 minutes then flip ... let rest at least 10 minutes", "The roasting sequence and resting time match passage 1."),
    (("17577_c02", "17577_c03", "17577_c04", "17577_c05", "17577_c06", "17577_c07"), ["passage_1", "passage_2", "passage_3"], "MobileGo ... backup and restore ... Connect the device to a computer ... pictures ... LG G3 ... PC", "The three passages respectively cover a manager, generic transfer and LG-G3-to-PC transfer; the absence qualifications are source-bounded."),
    (("17621_c03", "17621_c04", "17621_c05"), ["passage_2", "passage_3"], "dampened washcloth ... coarse-grit sandpaper ... sand in the direction of the grain", "The cabinet preparation and sanding steps match the source."),
    (("17690_c02", "17690_c08", "17690_c09", "17690_c10"), ["passage_1", "passage_2", "passage_3"], "soak for about 40 hours ... separate ... spent grain ... wort ... huge brew kettle", "The malting, lautering and boiling steps are stated in the brewing passages."),
    (("17696_c02", "17696_c03", "17696_c04", "17696_c05", "17696_c06"), ["passage_2", "passage_3"], "Preheat the grill to medium-high ... Brush ... olive oil ... aluminum foil square ... one tablespoon of butter", "The preparation and foil-placement steps appear in the zucchini source."),
]

for _ids, _evidence_ids, _quote, _reason in MANUAL_KEEP_GROUPS:
    for _pid in _ids:
        if _pid in REVIEWED:
            raise ValueError(f"Duplicate reviewed pair {_pid}")
        REVIEWED[_pid] = {
            "eligibility": "eligible", "audited_gold_label": "supported", "decision": "keep",
            "evidence_ids": _evidence_ids, "evidence_quote": _quote,
            "review_reason": _reason, "suggested_claims": [],
        }

# Evidence-absence decisions were made after reading all three passages in
# each source group. They are not claims about general world knowledge.
ABSENCE_REVIEW = {
    "12639_c13": "No passage mentions required minimum distributions during the Roth IRA holder's lifetime.",
    "12639_c15": "No passage discusses Social Security taxes on Roth IRA withdrawals.",
    "12639_c17": "No passage discusses Medicare taxes on Roth IRA withdrawals.",
    "12639_c19": "No passage discusses income limits for Roth IRA contributions.",
    "12639_c21": "No passage discusses recontributing withdrawn Roth IRA funds within three years.",
    "12639_c23": "No passage discusses inheritance tax consequences for Roth IRAs.",
    "12675_c05": "The sources describe sorbet ingredients and sherbet texture, but not the asserted sorbet and gelato texture comparison.",
    "12837_c05": "The DEA passages do not state a 90-day submission deadline.",
    "12837_c06": "The DEA passages do not list the claimed marriage, discharge and income documents.",
    "13030_c03": "The sources mention chord inversions as a learner topic but do not establish the claimed general guitar inversion rule.",
    "13030_c04": "No passage instructs the learner to choose an inversion to fit a desired sound.",
    "15605_c04": "The sources give retirement contribution details, not the contents of Penn State Human Resources' website.",
    "15605_c05": "The sources do not direct readers to a Human Resources website retirement-benefits section.",
    "15605_c06": "The sources do not describe a Penn State retirement-plan document's contents.",
    "15605_c07": "The sources do not say a plan document can be requested from Human Resources or accessed online.",
    "15605_c08": "The sources do not identify TIAA as the Penn State plan provider.",
    "15605_c09": "The sources do not identify a TIAA website with Penn State plan options.",
    "16287_c04": "The supplied verses include John 8:12 and John 10:9, but not the claimed John 8:24 quotation.",
    "16287_c06": "The supplied verses do not contain the claimed John 14:6 quotation.",
    "16287_c07": "The supplied verses do not contain the claimed John 15:1–5 quotation.",
    "16287_c08": "The supplied verses do not contain the claimed John 17:3 quotation.",
    "17014_c06": "The cat-grass passages do not provide the claimed seed depth or spacing.",
    "17621_c13": "The cabinet-painting passages do not state ventilation, gloves or goggles guidance.",
    "17621_c07": "The passages say to apply primer but do not state that it must dry completely before painting.",
    "17690_c03": "The brewing passages mention soaking grain, but not germination, enzymes or starch breakdown at mashing.",
    "17690_c04": "Milling appears in the brewing-stage list, but its process and starchy endosperm details are absent.",
    "17690_c05": "The sources do not explain enzymes accessing starch or breaking it into simple sugars.",
    "17690_c06": "Mashing is named, but the asserted hot-water mixing and mash-tun vessel are not described.",
    "17690_c07": "The sources do not describe starch gelatinization or conversion to alcohol and carbon dioxide at mashing.",
    "17690_c12": "The sources do not state that boiling sterilizes wort or kills bacteria.",
    "17690_c13": "Cooling is named in a stage list, but its after-boiling procedure and temperature target are not stated.",
    "17690_c14": "The sources do not describe transfer to an insulated fermentation tank during cooling.",
    "17690_c16": "Ale yeast fermentation is described, but the stated sugar-to-alcohol-and-CO2 reaction is not.",
    "17690_c18": "The sources list racking and finishing without saying residual yeast or solids are removed.",
    "17696_c08": "The zucchini passages provide direct grilling steps and foil-pack ingredients, but not a 10–15 minute foil-pack cooking time.",
    "15015_c02": "The source establishes that lenders work with delinquent borrowers, but does not specify a separate identification-of-delinquency procedure.",
    "13030_c09": "The sources do not state that conversion varies with the desired interpretation of individual progressions.",
}

for _pid, _referent in {
    "16287_c12": "Son of God (John 8:24)", "16287_c14": "Way, truth, and life (John 14:6)",
    "16287_c15": "Vine (John 15:1-5)", "16287_c16": "Son of Man (John 17:3)",
}.items():
    REVIEWED[_pid] = {
        "eligibility": "needs_context", "audited_gold_label": "needs_adjudication",
        "decision": "review_and_split", "evidence_ids": [], "evidence_quote": None,
        "review_reason": f"The fragment '{_referent}' lacks a standalone predicate; recover the preceding list heading and source verse before judging the quotation relation.",
        "suggested_claims": [],
    }


def nonclaim_reason(claim: str) -> str:
    if claim.startswith("(Passage") or (claim.endswith(")") and claim[:-1].isdigit()):
        return "Citation marker was split into a standalone pair; it expresses no verifiable claim."
    if claim.lower().startswith(("sure", "i hope", "let me", "if none of these options")):
        return "Conversation acknowledgement or closing, without a verifiable proposition."
    if "unable to answer" in claim.lower() or "cannot answer" in claim.lower() or "cannot provide a brief answer" in claim.lower():
        return "Refusal statement, not an answer fact to check against source evidence."
    if claim.endswith(":") or "follow these steps" in claim.lower():
        return "List introduction or section heading; the verifiable content is in following claims."
    return "Response framing without a standalone verifiable relation."


def main() -> None:
    if DEST.exists() or MANIFEST.exists():
        raise FileExistsError("Audit draft already exists; preserve prior decisions and choose a new version")
    pairs = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {p["pair_id"]: p for p in pairs}
    if len(pairs) != 414 or len(by_id) != len(pairs):
        raise ValueError("Expected 414 distinct pair IDs")
    unknown = (NONCLAIM | REVIEWED.keys() | ABSENCE_REVIEW.keys()) - by_id.keys()
    if unknown or NONCLAIM & (REVIEWED.keys() | ABSENCE_REVIEW.keys()) or REVIEWED.keys() & ABSENCE_REVIEW.keys():
        raise ValueError(f"Invalid manual IDs: {sorted(unknown)}")
    rows = []
    for p in pairs:
        pid = p["pair_id"]
        row = {
            "pair_id": pid, "qid": p["qid"], "source_id": p["source_id"],
            "claim_text": p["claim"], "response_offsets": [p["claim_start"], p["claim_end"]],
            "original_gold_label": p["gold_label"], "raw_span_labels": p["gold_label_raw"],
            "evidence_ids": [], "evidence_quote": None, "suggested_claims": [],
            "label_version": VERSION,
        }
        if pid in NONCLAIM:
            row.update(eligibility="nonclaim", audited_gold_label="not_applicable",
                       decision="exclude_from_claim_metrics", review_reason=nonclaim_reason(p["claim"]))
        elif pid in REVIEWED:
            row.update(REVIEWED[pid])
        elif pid in ABSENCE_REVIEW:
            row.update(eligibility="eligible", audited_gold_label="unsupported",
                       decision="keep" if p["gold_label"] == "unsupported" else "relabel",
                       review_reason="Inspected all three source passages. " + ABSENCE_REVIEW[pid])
        else:
            row.update(eligibility=None, audited_gold_label=None, decision=None,
                       review_reason="Pending source-level manual review")
        reviewed = pid in NONCLAIM or pid in REVIEWED or pid in ABSENCE_REVIEW
        row.update(auditor=AUDITOR if reviewed else None,
                   audited_at=date.today().isoformat() if reviewed else None,
                   audit_status="reviewed" if reviewed else "pending")
        rows.append(row)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    counts = Counter((row["audit_status"], row["eligibility"]) for row in rows)
    manifest = {
        "status": "complete_pending_researcher_confirmation", "label_version": VERSION,
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "audit_path": str(DEST.relative_to(ROOT)),
        "audit_sha256": hashlib.sha256(DEST.read_bytes()).hexdigest(),
        "pair_count": len(rows), "reviewed_count": sum(r["audit_status"] == "reviewed" for r in rows),
        "pending_count": sum(r["audit_status"] == "pending" for r in rows),
        "category_counts": {f"{status}:{eligibility}": count for (status, eligibility), count in sorted(counts.items(), key=lambda item: str(item[0]))},
        "notes": "Explicit reviewed decisions only. Pending rows are not audited gold. Researcher confirmation is required before freezing an evaluation set.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
