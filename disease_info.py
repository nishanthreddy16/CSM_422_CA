"""
Simple knowledge base: gives a description + treatment tips for a predicted class.
It matches keywords in the class name, so it works for any PlantVillage class.
"""

GENERAL_TIPS = ("The suggestions shown are general guidance only, not a confirmed diagnosis. "
                "Confirm with a local agriculture officer before spraying chemicals.")

_RULES = [
    ("healthy", {
        "status": "Healthy",
        "description": "The leaf shows no visible signs of disease.",
        "treatment": "No treatment needed.",
        "prevention": "Keep regular watering, balanced fertiliser and good spacing between plants.",
    }),
    ("virus", {
        "status": "Diseased",
        "description": "A viral disease that causes mottled, curled or yellowed leaves. It spreads through insects such as whiteflies and aphids.",
        "treatment": "No cure exists. Remove and destroy infected plants. Control insect carriers.",
        "prevention": "Use virus-free seeds, control whiteflies/aphids, and disinfect tools.",
    }),
    ("mosaic", {
        "status": "Diseased",
        "description": "A mosaic virus that creates light and dark green patches on leaves.",
        "treatment": "Remove infected plants. There is no chemical cure.",
        "prevention": "Use resistant varieties and wash hands and tools after touching infected plants.",
    }),
    ("curl", {
        "status": "Diseased",
        "description": "Leaf-curl virus spread by whiteflies; leaves curl upward and turn yellow.",
        "treatment": "Remove infected plants and control whiteflies.",
        "prevention": "Use resistant varieties, yellow sticky traps and insect nets.",
    }),
    ("late_blight", {
        "status": "Diseased",
        "description": "Late blight is a fast-spreading fungus-like disease that creates dark, water-soaked patches on leaves and fruit, especially in cool and wet weather.",
        "treatment": "Remove infected leaves immediately. Apply a copper-based or recommended fungicide.",
        "prevention": "Avoid overhead watering, improve air flow and rotate crops every season.",
    }),
    ("blight", {
        "status": "Diseased",
        "description": "Blight causes brown spots, often with rings, which grow and make leaves dry and fall.",
        "treatment": "Remove affected leaves and apply a suitable fungicide.",
        "prevention": "Water at the base of the plant, rotate crops and remove old plant debris.",
    }),
    ("rust", {
        "status": "Diseased",
        "description": "Rust fungus forms orange or reddish-brown powdery pustules on leaves.",
        "treatment": "Remove infected leaves and apply a sulfur or fungicide spray.",
        "prevention": "Avoid wet leaves, give plants space and use resistant varieties.",
    }),
    ("mildew", {
        "status": "Diseased",
        "description": "Powdery mildew appears as a white powder-like coating on leaves.",
        "treatment": "Spray sulfur-based or neem-oil fungicide and prune crowded leaves.",
        "prevention": "Improve air circulation and avoid excess nitrogen fertiliser.",
    }),
    ("scab", {
        "status": "Diseased",
        "description": "Scab fungus causes dark, rough, scabby spots on leaves and fruit.",
        "treatment": "Remove fallen leaves and apply a fungicide early in the season.",
        "prevention": "Clean up fallen leaves in autumn and prune for good air flow.",
    }),
    ("rot", {
        "status": "Diseased",
        "description": "A rot disease that causes dark, decaying patches on leaves and fruit.",
        "treatment": "Prune and destroy infected parts and apply a recommended fungicide.",
        "prevention": "Remove mummified fruit, avoid wet foliage and keep the field clean.",
    }),
    ("mold", {
        "status": "Diseased",
        "description": "Leaf mold shows pale yellow spots on top of the leaf and a grey-olive mold below.",
        "treatment": "Reduce humidity, remove infected leaves and apply a fungicide.",
        "prevention": "Improve ventilation and avoid watering the leaves.",
    }),
    ("mite", {
        "status": "Diseased",
        "description": "Spider mites are tiny pests that cause yellow speckles and fine webbing on leaves.",
        "treatment": "Spray insecticidal soap, neem oil or a miticide. Wash leaves with water.",
        "prevention": "Keep plants well watered and check the undersides of leaves regularly.",
    }),
    ("scorch", {
        "status": "Diseased",
        "description": "Leaf scorch causes brown, dry edges that look burnt.",
        "treatment": "Remove damaged leaves, water deeply and treat if caused by a fungus.",
        "prevention": "Avoid water stress and extreme heat exposure.",
    }),
    ("greening", {
        "status": "Diseased",
        "description": "Citrus greening (HLB) causes yellow shoots, blotchy leaves and bitter fruit. It is spread by psyllid insects.",
        "treatment": "No cure. Remove infected trees and control psyllids.",
        "prevention": "Use certified disease-free plants and control psyllid insects.",
    }),
    ("spot", {
        "status": "Diseased",
        "description": "A leaf-spot disease that creates dark or brown spots, sometimes with yellow rings, which reduce photosynthesis.",
        "treatment": "Remove spotted leaves and apply a copper or other recommended fungicide/bactericide.",
        "prevention": "Avoid overhead watering, rotate crops and keep tools clean.",
    }),
]

_UNKNOWN = {
    "status": "Unknown",
    "description": "No stored information for this class.",
    "treatment": "Consult a local agriculture expert.",
    "prevention": "Maintain good field hygiene and crop rotation.",
}


def get_info(class_name):
    """Return a dict with status, description, treatment and prevention."""
    key = class_name.lower()
    for keyword, info in _RULES:
        if keyword in key:
            return info
    return _UNKNOWN
