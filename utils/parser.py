import json
import os


def parse_input(json_path: str):
    with open(json_path) as json_file:
        input = json.load(json_file)
    return input


def write_output(predictions):
    os.makedirs(os.path.join('output'), exist_ok=True)
    with open(os.path.join('output', 'predictions.json'), 'w') as json_file:
        json.dump(predictions, json_file)



#
# "wind": {
#     "00h": 5,
# }
#
#
# input = {
#         "wind": {
#             "00h": {
#                 "wind_speed": 3.4
#             },
#             "06h": {
#                 "wind_speed": 2.4
#             },
#             "12h": {
#                 "wind_speed": 8.4
#             },
#             "18h": {
#                 "wind_speed": 0.0
#             }
#         },
#         "solar": {
#             "00h": {
#                 "irradiance_direct": 0.4,
#                 "irradiance_diffuse": 0.5
#             },
#             "06h": {
#                 "irradiance_direct": 0.3,
#                 "irradiance_diffuse": 0.8
#             },
#             "12h": {
#                 "irradiance_direct": 0.0,
#                 "irradiance_diffuse": 0.1
#             },
#             "18h": {
#                 "irradiance_direct": 0.1,
#                 "irradiance_diffuse": 0.2
#             }
#         }
#     }
#     with open(os.path.join('input', 'input_example.json'), 'w') as outfile:
#         json.dump(input, outfile)
#
