from langgraph.graph.state import StateGraph
from langgraph.constants import  START, END
from typing import TypedDict, Tuple, Optional
from model import predict_input_ev, predict_input_ice
from agents.feedback import get_feedback_chain

class EcoDrivingData(TypedDict):
    "avg_speed_kmph": float,
    "top_speed_kmph": float,
    "acceleration_aggressiveness": int,
    "braking_hard_events": int,
    "idle_time_min": float,
    "terrain_type": str,
    "ambient_temperature_C": float,
    "ac_usage_percent": int,
    "regen_braking_enabled": int,
    "eco_mode_enabled": int,
    "battery_usage_kWh": float

def predict_input_ev(single_input: EcoDrivingData) -> str:
    """
    Predicts the eco-driving score for an EV based on input data.
    """
    return predict_input_ev(single_input)

def predict_input_ice(single_input: EcoDrivingData) -> str:
    """
    Predicts the eco-driving score for an ICE vehicle based on input data.
    """
    return predict_input_ice(single_input)  

def get_feedback_chain(res_ev: str, res_ice: str) -> Tuple[str, Optional[str]]:
    """
    Generates feedback based on the predictions for EV and ICE vehicles.
    """
    return get_feedback_chain(res_ev, res_ice)  

def vehicle_type(vehicle_data: EcoDrivingData) -> str:
    """
    Determines the type of vehicle based on the input data.
    """
    if "" :
        return "EV"
    else:
        return "ICE"

graph = StateGraph(EcoDrivingData)
graph.add_node("EV", predict_input_ev)
graph.add_node("ICE", predict_input_ice)
graph.add_node("router", lambda vehicle_type: vehicle_type)



graph.add_node("FeedBack", get_feedback_chain)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    vehicle_type,, 

    {
        # Edge: Node
        "EV": "predict_input_ev",
        "ICE": "predict_input_ice"
    }

)
graph.add_edge("predict_input_ev", "FeedBack")
graph.add_edge("predict_input_ice", "FeedBack")

graph.add_edge("FeedBack", END)

EVROUTE_APP = graph.compile()