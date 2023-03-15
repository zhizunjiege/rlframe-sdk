#include <any>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

/*
 * This function is called in every simulation step with a map of states, where the key is the name of the model and the value is a vector of entities.
 * Each entity in the vector represents a sets of output params, and the key is the name of the param and the value is wrapped in std::any.
 * The function should return `true` if the simulation should be terminated.
 */
bool func(std::unordered_map<std::string, std::vector<std::unordered_map<std::string, std::any>>> &states)
{
  auto &uav = states["example_uav"][0];
  auto &sub = states["example_sub"][0];
  auto uav_lon = std::any_cast<double>(uav["longitude"]);
  auto uav_lat = std::any_cast<double>(uav["latitude"]);
  auto sub_lon = std::any_cast<double>(sub["longitude"]);
  auto sub_lat = std::any_cast<double>(sub["latitude"]);
  auto dist = std::sqrt(std::pow(uav_lon - sub_lon, 2) + std::pow(uav_lat - sub_lat, 2));
  return dist <= 0.1;
}
