# [**비자동회귀 다중 디코더 기반 한국어 형태소 분석**](https://koreascience.kr/article/CFKO202226455347146.page)

  <!-- | Name  | Type          | Params
----------------------------------------
0 | model | NATransformer | 35.1 M
----------------------------------------
35.1 M    Trainable params
0         Non-trainable params
35.1 M    Total params
140.378   Total estimated model params size (MB) -->




## 학습

```
bash run_trian.sh
```


## 추론

```
bash inference.sh
```


## 실험 결과

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>형태소 F1</th>
            <th>어절 Acc</th>
            <th>시간 </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td >자동회귀 트랜스포머</td>
            <td > 97.72  </td>
            <td> 95.85     </td>
            <td> 1.00x      </td>
        </tr>
        <tr>
            <td >제안 방법</td>
            <td > 97.59 </td>
            <td> 95.83 </td>
            <td> 10.05x      </td>
        </tr>
    </tbody>
</table>



