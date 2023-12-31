﻿// Copyright (c) Damir Dobric. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.
using NeoCortexApi.Entities;
using System;
using System.Collections;
using System.Collections.Generic;

namespace NeoCortexApi.Utility
{
    /// <summary>
    /// Allegory to the Python itertools.groupby. Objects of this class take a list of inputs and a function to produce keys. The iterator
    /// or foreach loop generates grouped return values based on the key generated by the supplied function.<br></br>
    /// For instance:<br></br>
    /// Given the list:<br></br>
    /// List&lt;Integer&gt; list = new List&lt;Integer&gt;() { 2, 4, 4, 5 };<br></br>
    /// and the function:<br></br>
    /// Func&lt;Integer, Integer&gt; lambda = x => x * 3;<para/>
    /// 
    /// A GroupBy can be compose as such:<br></br>
    /// GroupBy&lt;Integer, Integer> grouper = GroupBy.From(l, lambda);<para/>
    /// 
    /// ...then iterated over as such:<br></br>
    /// foreach (Pair&lt;Integer, Integer> p in grouper) <br></br>
    /// {<br></br>
    ///     Console.WriteLine($"Pair key: {p.Key}, pair value: {p.Value}");<br></br>
    /// }<para/>
    /// 
    /// Outputs:<para/>
    /// 
    /// Pair key: 2, pair value: 6<br></br>
    /// Pair key: 4, pair value: 12<br></br>
    /// Pair key: 4, pair value: 12<br></br>
    /// Pair key: 5, pair value: 15<br></br>
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="R"></typeparam>
    /// <remarks>
    /// Notes: Read up on groupby <a href="https://docs.python.org/dev/library/itertools.html#itertools.groupby">here</a> 
    /// </remarks>
    public class GroupBy<T, R> : IEnumerable<Pair<T, R>>, IEnumerator<Pair<T, R>>
    {

        /** serial version */
        //private static readonly long serialVersionUID = 1L;

        private bool m_IsStarted;

        private List<T> m_ElementList;
        private Func<T, R> m_Func;
        private IntGenerator m_IntegerGenerator;
        private Pair<T, R> m_CurrentElement;


        public Pair<T, R> Current => this.m_CurrentElement;

        object IEnumerator.Current => this.m_CurrentElement;


        #region Constructors and Initialization


        /**
         * Constructs a new {@code GroupBy}
         * 
         * @param l     the {@link List} containing the items used as input to the
         *              key generating function.     
         * @param fn    the {@link Function} to be used to generate the keys which describe
         *              the like contents of each grouping.
         */
        /// <summary>
        /// Constructs a new <see cref="GroupBy{T, R}"/>
        /// </summary>
        /// <param name="l">the <see cref="List{T}"/> containing the items used as input to the key generating function.</param>
        /// <param name="func">the <see cref="Func{T, TResult}"/> to be used to generate the keys which describe the like contents of each grouping</param>
        public GroupBy(List<T> l, Func<T, R> func)
        {
            this.m_ElementList = l;
            this.m_Func = func;
            this.m_IntegerGenerator = IntGenerator.Of(0, m_ElementList.Count);

            if (m_IntegerGenerator.HasNext())
            {
                T t = m_ElementList[m_IntegerGenerator.Get()];
                m_IsStarted = false;
                m_CurrentElement = new Pair<T, R>(t, func(t));
            }
        }


        /**
         * Returns a new {@code GroupBy} composed from the specified list 
         * and key-generating {@link Function}
         * 
         * @param l     the {@link List} containing the items used as input to the
         *              key generating function.     
         * @param fn    the {@link Function} to be used to generate the keys which describe
         *              the like contents of each grouping.
         * @return
         */
        /// <summary>
        /// Returns a new <see cref="GroupBy{T, R}"/> composed from the specified list and key-generating <see cref="Func{T, TResult}"/>
        /// </summary>
        /// <param name="l">the <see cref="List{T}"/> containing the items used as input to the key generating function.</param>
        /// <param name="fn">the <see cref="Func{T, TResult}"/> to be used to generate the keys which describe the like contents of each grouping</param>
        /// <returns></returns>
        public static GroupBy<T, R> From(List<T> l, Func<T, R> fn)
        {
            return new GroupBy<T, R>(l, fn);
        }
        #endregion


        /**
         * {@inheritDoc}
         */
        // @Override

        [Obsolete("Use .Current instead")]
        public Pair<T, R> peek()
        {
            return m_CurrentElement;
        }

        /**
         * {@inheritDoc}
         */
        // @Override
        //public bool hasNext()
        //{
        //    return m_CurrentElement != null;
        //}


        /// <summary>
        /// Moves to the nex pair.
        /// </summary>
        /// <returns></returns>
        public bool MoveNext()
        {
            Pair<T, R> ret = m_CurrentElement;

            if (m_IntegerGenerator.HasNext())
            {
                T t;

                if (m_IsStarted)
                {
                    m_IntegerGenerator.Next();
                }
                else
                {
                    m_IsStarted = true;
                }

                t = m_ElementList[m_IntegerGenerator.Get()];

                m_CurrentElement = new Pair<T, R>(t, m_Func(t));

                return true;
            }
            else
            {
                m_CurrentElement = null;
                return false;
            }
        }

        /// <summary>
        /// Shows the next pair, but it does not move internal pointer to it.
        /// </summary>
        /// <returns></returns>
        public Pair<T, R> NextPair
        {
            get
            {
                Pair<T, R> ret;

                if (m_IntegerGenerator.HasNext())
                {
                    var nextVal = m_IntegerGenerator.NextValue;

                    T t = m_ElementList[nextVal];

                    ret = new Pair<T, R>(t, m_Func(t));
                }
                else
                {
                    ret = null;
                }

                return ret;
            }
        }

        public void Reset()
        {
            m_IntegerGenerator.Reset();
            m_IntegerGenerator.Next();
            m_CurrentElement = null;
            m_IsStarted = false;
        }

        public void Dispose()
        {

        }

        public IEnumerator<Pair<T, R>> GetEnumerator()
        {
            return this;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this;
        }
    }

}
