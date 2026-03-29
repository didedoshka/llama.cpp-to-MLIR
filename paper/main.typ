#import "template.typ": *

#show: term_paper

#generate_title()

#generate_outline()

#show math.ast: math.dot.c


= Аннотация

Развитие моделей машинного обучения и увеличение их размеров требует значительных вычислительных ресурсов для их эффективного выполнения. По этой причине в мире повысился интерес к разработке специализированных процессоров для ускорения выполнения нейронных сетей. Эффективное использование всех доступных возможностей целевого аппаратного обеспечения невозможно без развитого стека программного обеспечения. По этой причине исследования в области построения оптимизирующих компиляторов нейронных сетей является критически важными и актуальными. На сегодняшний день формат GGUF и библиотека llama.cpp стали промышленным стандартом для эффективного выполнения больших языковых моделей на локальных устройствах, в особенности на устройствах с ограниченными вычислительными ресурсами. К числу последних можно отнести встраиваемые системы, устройства, работающие под управлением энерго-эффективных процессоров. Однако этот эта библиотека не выполняет оптимизирующих трансформаций вычислительного графа, а только выполняет его при помощи вручную созданных kernel-функций.
В данной работе предлагается разработать компилятор на базе инфраструктуры MLIR. Разрабатываемый компилятор будет выполнять трансляцию операций вычислительного графа проекта GGML (подпроект llama.cpp) в операции линейной алгебры, реализованные в виде MLIR-диалекта linalg. Это позволит проводить аппаратно-независимые и архитектурно-ориентированные оптимизации операций линейной алгебры. В рамках проекта, также планируется поддержать некоторые алгоритмы квантования, используемых в инфраструктуре llama.cpp.


= Введение в MLIR

== Для чего нужен MLIR (https://mlir.llvm.org/) //TODO: на сайте есть информация, как цитировать
MLIR --- гибридное промежуточное представление (IR), разработанное с целью удовлетворить следующие требования:
// MLIR is intended to be a hybrid IR which can support multiple different requirements in a unified infrastructure. For example, this includes:
- Возможность задавать графы потока данных (такие как TensorFlow (https://www.tensorflow.org/)), включая динамические размерности тензоров, пользовательские операции, переменные TensorFlow, и т. д.
// The ability to represent dataflow graphs (such as in TensorFlow), including dynamic shapes, the user-extensible op ecosystem, TensorFlow variables, etc.
- Возможность реализации оптимизаций и преобразований, типично применяемых к таким графам (например, в Grappler (https://www.tensorflow.org/guide/graph_optimization))
// Optimizations and transformations typically done on such graphs (e.g. in Grappler).
- Возможность поддерживать оптимизации циклов для высокопроизводительных вычислений (слияние (fusion), (перестановка циклов) loop interchange, tiling, и т. д.)
// Ability to host high-performance-computing-style loop optimizations across kernels (fusion, loop interchange, tiling, etc.), and to transform memory layouts of data.
- Возможность реализации "понижающих" преобразований кодогенерации, таких как вставка операций прямого обращения к памяти (DMA insertion), явное управление кэшем (explicit cache management), memory tiling и векторизация для архитектур с 1D и 2D регистрами
// Code generation “lowering” transformations such as DMA insertion, explicit cache management, memory tiling, and vectorization for 1D and 2D register architectures.
- Возможность использовать операции конкретных архитектур, например высокоуровневые операции конкретного ускорителя
// Ability to represent target-specific operations, e.g. accelerator-specific high-level operations.
- Возможносте поддерживать квантование и другие трансформации, применяемые к вычислительным графам глубинного обучения
// Quantization and other graph transformations done on a Deep-Learning graph.
- Возможность иметь примитивы выпуклых многогранников (polyhedral)
// Polyhedral primitives.
- Hardware Synthesis Tools / HLS (https://en.wikipedia.org/wiki/High-level_synthesis)
// Hardware Synthesis Tools / HLS.

MLIR это общее промежуточное представление, которое также поддерживает операции конкретных архитектур. Таким образом инвестиции в инфраструктуру вокруг MLIR (например компиляторные проходы) должны возвращать хорошие результаты; многие целевые платформы могут использовать эту инфраструктуру и получать от этого пользу.
// MLIR is a common IR that also supports hardware specific operations. Thus, any investment into the infrastructure surrounding MLIR (e.g. the compiler passes that work on it) should yield good returns; many targets can use that infrastructure and will benefit from it.

MLIR это мощное представление, но существуют задачи, поддержка которых не предусмотрена. MLIR не предполагается использовать для реализации алгоритмов генерации машинного кода (таких как распределение регистров (register allocation), переупорядочевание инструкций (instruction scheduling)). Для них больше подходят оптимизирующие компиляторы низкого уровня (такие как LLVM). Также, MLIR не предполагается языком, на котором конечные пользователи будут писать код (аналогично CUDA C++). С другой стороны, MLIR предоставляет основу для реализации любого предметно-ориентированного языка (https://en.wikipedia.org/wiki/Domain-specific_language) и его интеграции в экосистему
// MLIR is a powerful representation, but it also has non-goals. We do not try to support low level machine code generation algorithms (like register allocation and instruction scheduling). They are a better fit for lower level optimizers (such as LLVM). Also, we do not intend MLIR to be a source language that end-users would themselves write kernels in (analogous to CUDA C++). On the other hand, MLIR provides the backbone for representing any such DSL and integrating it in the ecosystem.


= Квантование
//TODO: добавить цитату, например https://huggingface.co/docs/optimum/en/concept_guides/quantization или https://huggingface.co/docs/hub/en/gguf
Количество параметров в моделях машинного обучения, а особенно в больших языковых моделях, постоянно растет. В настоящее время распространены модели с десятками, и даже сотнями миллиардов параметров. В свою очередь объемы оперативной памяти (RAM) и памяти ускорителей (VRAM) увеличиваются недостаточно быстро. Для решения данной проблемы было предложено квантование: сжатие параметров, влекущее за собой потерю точности в обмен на уменьшение требуемой памяти. В ходе экспериментов выяснилось, что значительное понижение используемой памяти влечет намного менее значительное понижение качества модели. Особенно необходимо квантование в условиях крайней ограниченности доступной памяти.

Описание стандартных типов: Обычно модели используют или числа одинарной точности(float32, F32), или числа половинной точности (float16, F16). Целые числа любой ширины: int_k(I_k) может принимать $2^k$ различных значений
// добавить цитату (https://en.wikipedia.org/wiki/Half-precision_floating-point_format)

== Простейшие техники квантования
//TODO: написать в итоге сколько бит на число
Рассматривается блок чисел (размером 32 в GGML, https://github.com/ggml-org/ggml/blob/a8db410a252c8c8f2d120c6f2e7133ebe032f35d/src/ggml-common.h#L170)
=== Симметричное квантование (type-0 в GGML https://github.com/ggml-org/ggml/blob/a8db410a252c8c8f2d120c6f2e7133ebe032f35d/src/ggml-quants.c#L36)
Дан блок чисел $w_i$. Хотим для него хранить float $d$ (коэффициент масштабирования) и массив квантов int_k $q_i$, так что $w_i = d * q_i$. 

Пусть $a = max(abs(w_i))$. Получается мы хотим сжать отрезок $[-a, a]$ в отрезок $[-2^(k - 1), 2^(k - 1)]$. Таким образом в качестве $d$ следует взять $a / 2^(k - 1)$. Кванты вычисляются по формуле $q_i = round(w_i / d)$
=== Аффинное квантование (type-1 в GGML https://github.com/ggml-org/ggml/blob/a8db410a252c8c8f2d120c6f2e7133ebe032f35d/src/ggml-quants.c#L73)
Дан блок чисел $w_i$. Хотим для него хранить два float: $d$ (коэффициент масштабирования) и $m$ (минимум), а также массив квантов int_k $q_i$, так что $w_i = m + d * q_i$. 

Пусть $m = min(w_i)$, а $a = max(w_i)$. Получается мы хотим сжать отрезок $[m, a]$ в отрезок $[0, 2^k]$. Таким образом в качестве $d$ следует взять $(a - m) / 2^k$. Кванты вычисляются по формуле $q_i = round((w_i - m) / d)$

== K-кванты (https://github.com/ggml-org/llama.cpp/pull/1684#issue-1739619305)
https://github.com/ggml-org/ggml/blob/a8db410a252c8c8f2d120c6f2e7133ebe032f35d/src/ggml-quants.c#L1692

Используются супер-блоки. Каждый супер-блок содержит 16 блоков, каждый из которых содержит 16 чисел. В зависимости от метода для каждого блока выполняется или аффинное, или симметричное квантование, после чего их коэффициенты масштабирования ($d$) и минимумы ($m$) квантуются.
// Код квантования сложнее, надо разобраться и описать что именно происходит. Там тоже доступна importance matrix
//TODO: табличка

== Importance matrix (IQ2_XXS и т. д.)
https://github.com/ggml-org/llama.cpp/pull/4773 IQ2_XXS
https://github.com/ggml-org/llama.cpp/pull/4861 Importance matrix calculation
Идея заключается в том, что какие-то параметры важнее других. Подавая большое количество различных входных данных в модель можно вычислить Importance matrix --- матрицу важности каждого скаляра. Используя эту информацию можно сжимать более эффективно, получать лучшее качество за меньшую память.
        // GGML_TYPE_IQ2_XXS = 16,
        // GGML_TYPE_IQ2_XS  = 17,
        // GGML_TYPE_IQ3_XXS = 18,
        // GGML_TYPE_IQ1_S   = 19,
        // GGML_TYPE_IQ4_NL  = 20,
        // GGML_TYPE_IQ3_S   = 21,
        // GGML_TYPE_IQ2_S   = 22,
        // GGML_TYPE_IQ4_XS  = 23,

