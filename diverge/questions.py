"""Question bank for the DIVERGE arena. Written by the DIVERGE team."""

QUESTIONS = [
    dict(id="q01", domain="AI · Energy", difficulty="Hard",
         text="What is the most underrated reason large AI models consume so much energy, and what would actually fix it?"),
    dict(id="q02", domain="Cities", difficulty="Medium",
         text="Why do cities that build more roads often end up with worse traffic, and what would actually reduce congestion?"),
    dict(id="q03", domain="Climate", difficulty="Hard",
         text="What is the biggest hidden bottleneck to decarbonizing the electricity grid?"),
    dict(id="q04", domain="Medicine", difficulty="Hard",
         text="Why might a medical treatment that works in clinical trials fail when deployed across a whole population?"),
    dict(id="q05", domain="Forecasting", difficulty="Medium",
         text="Why do experts often make worse forecasts than simple statistical models?"),
    dict(id="q06", domain="Economics", difficulty="Medium",
         text="Why do most startups fail, beyond simply running out of money?"),
    dict(id="q07", domain="Energy", difficulty="Medium",
         text="How could a city cut household energy use by 30% without any new technology?"),
    dict(id="q08", domain="Science", difficulty="Hard",
         text="What would happen to scientific progress if traditional peer review were abolished?"),
    dict(id="q09", domain="AI", difficulty="Hard",
         text="Why is it so hard to build AI systems that reliably admit when they are uncertain?"),
    dict(id="q10", domain="Education", difficulty="Medium",
         text="What single change would most improve learning outcomes in schools at near-zero cost?"),
]
BY_ID = {q["id"]: q for q in QUESTIONS}
