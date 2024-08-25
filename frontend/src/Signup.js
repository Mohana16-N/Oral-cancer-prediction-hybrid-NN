import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import signupValidation from './SignupValidation';

function Signup() {
    const [values, setValues] = useState({
        firstName: '',
        lastName: '',
        email: '',
        password: '',
        confirmPassword: ''
    });

    const navigate = useNavigate();
    const [errors, setErrors] = useState({});

    const handleInput = (event) => {
        setValues(prev => ({ ...prev, [event.target.name]: event.target.value }));
    }

    const handleSubmit = (event) => {
        event.preventDefault();
        const validationErrors = signupValidation(values);
        setErrors(validationErrors);

        if (Object.keys(validationErrors).length === 0) {
            axios.post('http://localhost:8081/signup', values)
                .then(res => {
                    if (res.status === 201) {
                        alert("Signup successfully");
                        navigate('/home');
                    } else {
                        alert("Signup failed");
                    }
                })
                .catch(err => {
                    console.log(err);
                    alert("Signup failed");
                });
        }
    }

    return (
        <div className='d-flex justify-content-center align-items-center bg-primary vh-100'>
            <div className='container'>
                <div className='bg-white p-5 rounded'>
                    <h1 className="text-center mb-4 fw-bold fs-1">Oral Cancer Prediction using CNN and RNN</h1>
                    <h2 className="text-center mb-4 fw-bold fs-2">Sign-Up</h2>
                    <form onSubmit={handleSubmit}>
                        <div className='mb-3'>
                            <label htmlFor="firstName" className="fs-5">Enter First Name:</label>
                            <input type="text" placeholder='First Name' name="firstName"
                                onChange={handleInput} className='form-control rounded-0 text-center fs-5' />
                            {errors.firstName && <span className="text-danger">{errors.firstName}</span>}
                        </div>
                        <div className='mb-3'>
                            <label htmlFor="lastName" className="fs-5">Enter Last Name:</label>
                            <input type="text" placeholder='Last Name' name="lastName"
                                onChange={handleInput} className='form-control rounded-0 text-center fs-5' />
                            {errors.lastName && <span className="text-danger">{errors.lastName}</span>}
                        </div>
                        <div className='mb-3'>
                            <label htmlFor="email" className="fs-5">Enter Email:</label>
                            <input type="email" placeholder='Email' name="email"
                                onChange={handleInput} className='form-control rounded-0 text-center fs-5' />
                            {errors.email && <span className="text-danger">{errors.email}</span>}
                        </div>
                        <div className='mb-3'>
                            <label htmlFor="password" className="fs-5">Enter Password:</label>
                            <input type="password" placeholder='Password' name="password"
                                onChange={handleInput} className='form-control rounded-0 text-center fs-5' />
                            {errors.password && <span className="text-danger">{errors.password}</span>}
                        </div>
                        <div className='mb-3'>
                            <label htmlFor="confirmPassword" className="fs-5">Confirm Password:</label>
                            <input type="password" placeholder='Re-enter Password' name="confirmPassword"
                                onChange={handleInput} className='form-control rounded-0 text-center fs-5' />
                            {errors.confirmPassword && <span className="text-danger">{errors.confirmPassword}</span>}
                        </div>
                        <button type="submit" className='btn btn-success w-100 fs-5'><strong>Sign up</strong></button>
                        <Link to='/signin' className='btn btn-default border w-100 bg-light rounded text-decoration-none fs-5'>Back to Login</Link>
                    </form>
                </div>
            </div>
        </div>
    );
}

export default Signup;
