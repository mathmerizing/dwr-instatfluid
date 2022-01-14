/**
 * @file   FluidDataPostprocessor.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-18, Stokes, UK
 * @date 2019-01-18, eG, UK
 * @date 2018-06-20, deal.II v9.0, UK
 * @date 2017-10-05, Biot, UK
 * @date 2015-08-26, DTM++/awave, UK
 * @date 2015-02-24, DTM++/ccfd, UK
 */

/*  Copyright (C) 2012-2019 by Uwe Koecher                                    */
/*                                                                            */
/*  This file is part of DTM++ .                                              */
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
/*  along with DTM++ .   If not, see <http://www.gnu.org/licenses/>.          */

// PROJECT includes
#include <fluid/FluidDataPostprocessor.tpl.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>

// C++ includes

namespace fluid {

template<int dim>
void
DataPostprocessor<dim>::
evaluate_vector_field(
	const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
	std::vector< dealii::Vector<double> > &postprocessed_quantities
) const {
	auto &xh = input_data.solution_values;
	
	Assert(
		xh.size() != 0,
		dealii::ExcEmptyObject()
	);
	
	const unsigned int n_q_points(xh.size());
	
	Assert(
		(xh[0].size() == (dim+1)), // b+p dimensions
		dealii::ExcInternalError()
	);
	
	Assert(
		postprocessed_quantities.size() == n_q_points,
		dealii::ExcInternalError()
	);
	
	Assert(
		postprocessed_quantities[0].size() == internal_n_components,
		dealii::ExcInternalError()
	);
	
	for (unsigned int q{0}; q < n_q_points; ++q) {
		////////////////////////////////////////////////////////////////////////
		// convection
		if (output_quantities & static_cast<unsigned int>(OutputQuantities::convection)) {
			for (unsigned int d{0}; d < dim; ++d) {
				postprocessed_quantities[q][internal_first_convection_component+d] =
					xh[q][first_convection_component+d];
			}
		}
		
		////////////////////////////////////////////////////////////////////////
		// store pressure
		if (output_quantities & static_cast<unsigned int>(OutputQuantities::pressure)) {
			postprocessed_quantities[q][internal_pressure_component] =
				xh[q][pressure_component];
		}
	}
}


template<int dim>
std::vector< std::string >
DataPostprocessor<dim>::
get_names() const {
	return postprocessed_quantities_names;
}


template<int dim>
std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation >
DataPostprocessor<dim>::
get_data_component_interpretation() const {
	return dci;
}


template<int dim>
dealii::UpdateFlags
DataPostprocessor<dim>::
get_needed_update_flags() const {
	return uflags;
}

} // namespace

#include "FluidDataPostprocessor.inst.in"
