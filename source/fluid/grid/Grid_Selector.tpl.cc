/**
 * @file Grid_Selector.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 *
 * @date 2021-11-05, Schaefer/Turek 2D, JR
 * @date 2019-11-11, UK
 * @date 2018-07-26, included from biot for dwr, UK
 * @date 2018-05-25, UK
 * @date 2016-02-12, UK
 */

/*  Copyright (C) 2012-2021 by Uwe Koecher and contributors                   */
/*                                                                            */
/*  This file is part of DTM++.                                               */
/*                                                                            */
/*  DTM++ is free software: you can redistribute it and/or modify             */
/*  it under the terms of the GNU Lesser General Public License as            */
/*  published by the Free Software Foundation, either                         */
/*  version 3 of the License, or (at your option) any later version.          */
/*                                                                            */
/*  DTM++ is distributed in the hope that it will be useful,                  */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*  GNU Lesser General Public License for more details.                       */
/*                                                                            */
/*  You should have received a copy of the GNU Lesser General Public License  */
/*  along with DTM++.   If not, see <http://www.gnu.org/licenses/>.           */

#include <DTM++/base/LogStream.hh>
#include <fluid/grid/Grid_Selector.tpl.hh>
#include <fluid/grid/Grids.hh>

namespace fluid {
namespace grid {

template <int dim>
void Selector<dim>::create_grid(
    std::shared_ptr<fluid::ParameterSet> parameter_set,
    std::shared_ptr<fluid::Grid<dim> > &grid) const {
  ////////////////////////////////////////////////////////////////////////////
  //
  auto Mesh_Boundary_Class{parameter_set->space.boundary.fluid.Grid_Class};
  DTM::pout << "grid selector: creating Mesh Boundary Class = "
            << Mesh_Boundary_Class << std::endl;

  // Grid_Class_Options
  auto _options{parameter_set->space.boundary.fluid.Grid_Class_Options};

  ////////////////////////////////////////////////////////////////////////////
  // parse the input string, arguments are splitted with spaces
  //
  std::string argument;
  std::vector<std::string> options;
  for (auto &character : _options) {
    if (!std::isspace(character) && (character != '\"')) {
      argument += character;
    } else {
      if (argument.size()) {
        options.push_back(argument);
        argument.clear();
      }
    }
  }

  if (argument.size()) {
    options.push_back(argument);
    argument.clear();
  }

  ////////////////////////////////////////////////////////////////////////////
  //
  DTM::pout << "* found configuration: mesh/grid class options = " << std::endl;
  for (auto &option : options) {
    DTM::pout << "\t" << option << std::endl;
  }
  DTM::pout << std::endl;

  DTM::pout << "* generating mesh" << std::endl;

  ////////////////////////////////////////////////////////////////////////////
  //
  if (!Mesh_Boundary_Class.compare(
          "Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow")) {
    AssertThrow(
        options.size() == 1,
        dealii::ExcMessage("Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow "
                           "options invalid, "
                           "please check your input file data."));

    grid = std::make_shared<
        fluid::grid::Grid_Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow<
            dim> >(parameter_set,
                   std::stod(options.at(0))  // y_out coordinate
    );

    DTM::pout << "fluid::grid selector: created "
              << "Grid_Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow"
              << ", with " << std::endl
              << "\ty_out = " << std::stod(options.at(0)) << " . " << std::endl
              << std::endl;

    return;
  }

  ////////////////////////////////////////////////////////////////////////////
  //
  if (!Mesh_Boundary_Class.compare(
          "Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow_2")) {
    AssertThrow(options.size() == 2,
                dealii::ExcMessage(
                    "Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow_2 "
                    "options invalid, "
                    "please check your input file data."));

    grid = std::make_shared<
        fluid::grid::Grid_Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow_2<
            dim> >(parameter_set,
                   std::stod(options.at(0)),  // x_in (inflow) coordinate
                   std::stod(options.at(1))   // x_out (outflow) coordinate
    );

    DTM::pout << "fluid::grid selector: created "
              << "Grid_Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow_2"
              << ", with " << std::endl
              << "\tx_in  = " << std::stod(options.at(0)) << " , " << std::endl
              << "\tx_out = " << std::stod(options.at(1)) << " . " << std::endl
              << std::endl;

    return;
  }

  ////////////////////////////////////////////////////////////////////////////
  //
  if (!Mesh_Boundary_Class.compare("Schaefer_Turek_2D")) {
    AssertThrow(options.size() == 0,
                dealii::ExcMessage("Schaefer_Turek_2D "
                                   "options invalid, "
                                   "please check your input file data."));

    grid = std::make_shared<fluid::grid::Grid_Schaefer_Turek_2D<dim> >(
        parameter_set);

    DTM::pout << "fluid::grid selector: created "
              << "Grid_Schaefer_Turek_2D" << std::endl;

    return;
  }
  ////////////////////////////////////////////////////////////////////////////
  //
  if (!Mesh_Boundary_Class.compare("Schaefer_Turek_3D")) {
    AssertThrow(options.size() == 0,
                dealii::ExcMessage("Schaefer_Turek_3D "
                                   "options invalid, "
                                   "please check your input file data."));

    grid = std::make_shared<fluid::grid::Grid_Schaefer_Turek_3D<dim> >(
        parameter_set);

    DTM::pout << "fluid::grid selector: created "
              << "Grid_Schaefer_Turek_3D" << std::endl;

    return;
  }
  ////////////////////////////////////////////////////////////////////////////
  //
  if (!Mesh_Boundary_Class.compare("Cavity_2D")) {
    AssertThrow(options.size() == 0,
                dealii::ExcMessage("Cavity_2D "
                                   "options invalid, "
                                   "please check your input file data."));

    grid = std::make_shared<fluid::grid::Grid_Cavity_2D<dim> >(parameter_set);

    DTM::pout << "fluid::grid selector: created "
              << "Grid_Cavity_2D" << std::endl;

    return;
  }

  ////////////////////////////////////////////////////////////////////////////
  //
  AssertThrow(false,
              dealii::ExcMessage(
                  "Grid Class unknown, please check your input file data."));
}

}  // namespace grid
}  // namespace fluid

#include "Grid_Selector.inst.in"
