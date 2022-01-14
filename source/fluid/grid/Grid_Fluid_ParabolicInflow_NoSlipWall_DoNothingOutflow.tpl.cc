/**
 * @file Grid_Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-11, UK
 * @date 2018-07-26, UK
 * @date 2018-03-06, UK
 */

/*  Copyright (C) 2012-2019 by Uwe Koecher and contributors                   */
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

// PROJECT includes
#include <fluid/grid/Grid_Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow.tpl.hh>
#include <fluid/types/boundary_id.hh>

namespace fluid {
namespace grid {

template<int dim>
void
Grid_Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow<dim>::
set_boundary_indicators() {
	// set boundary indicators (space)
	{
		auto slab(this->slabs.begin());
		auto ends(this->slabs.end());
		
		for (; slab != ends; ++slab) {
			auto cell(slab->space.tria->begin_active());
			auto endc(slab->space.tria->end());
			
			for (; cell != endc; ++cell) {
			if (cell->at_boundary()) {
			for (unsigned int face(0);
				face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
				if (cell->face(face)->at_boundary()) {
					
					auto center{cell->face(face)->center()};
					
					if ((std::abs(center[0]-(-.5)) < 1e-12)) {
						// inflow boundary
						cell->face(face)->set_boundary_id(
							static_cast<dealii::types::boundary_id> (
								fluid::types::space::boundary_id::prescribed_convection_c1 +
								fluid::types::space::boundary_id::prescribed_convection_c2 +
								fluid::types::space::boundary_id::prescribed_convection_c3
							)
						);
					}
					else if ((std::abs(center[0]-y_out) < 1e-12)) {
						// outflow boundary
						cell->face(face)->set_boundary_id(
							static_cast<dealii::types::boundary_id> (
								fluid::types::space::boundary_id::prescribed_do_nothing
							)
						);
					}
					else {
						// walls: no-flow / no-slip conditions:
						//   prescribed homog. Dirichlet for convection
						cell->face(face)->set_boundary_id(
							static_cast<dealii::types::boundary_id> (
								fluid::types::space::boundary_id::prescribed_no_slip
							)
						);
					}
					
				}
			}}}
		}
	}
	
	// set Sigma_0
	{
		if (this->slabs.size()) {
			auto slab_Q1(this->slabs.begin());
			
			auto cell(slab_Q1->time.tria->begin_active());
			auto endc(slab_Q1->time.tria->end());
			
			for (; cell != endc; ++cell) {
			if (cell->at_boundary()) {
			for (unsigned int face(0);
				face < dealii::GeometryInfo<1>::faces_per_cell; ++face) {
				if (cell->face(face)->at_boundary() && face==0) {
					
					cell->face(face)->set_boundary_id(
						static_cast<dealii::types::boundary_id> (
							fluid::types::time::boundary_id::Sigma0)
					);
				}
			}}}
		}
	}
	
	// set Sigma_T
	{
		if (this->slabs.size()) {
			auto slab_QN(this->slabs.rbegin());
			
			auto cell(slab_QN->time.tria->begin_active());
			auto endc(slab_QN->time.tria->end());
			
			for (; cell != endc; ++cell) {
			if (cell->at_boundary()) {
			for (unsigned int face(0);
				face < dealii::GeometryInfo<1>::faces_per_cell; ++face) {
				if (cell->face(face)->at_boundary() && face==1) {
					
					cell->face(face)->set_boundary_id(
						static_cast<dealii::types::boundary_id> (
							fluid::types::time::boundary_id::SigmaT)
					);
				}
			}}}
		}
	}
}

}} // namespaces

#include "Grid_Fluid_ParabolicInflow_NoSlipWall_DoNothingOutflow.inst.in"
